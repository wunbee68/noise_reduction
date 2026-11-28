import argparse
import csv
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from denoise.audio import CHALLENGE_SAMPLES
from denoise.data import manifest, read_pair
from denoise.fastenhancer import FastEnhancer
from denoise.metrics import pcm16, score
from denoise.refinement import SpectralRefiner


def load_data(root, split, weights, cache_dir):
    rows = manifest(root, split)
    weight_hash = sha256(weights.read_bytes()).hexdigest()
    manifest_hash = sha256((root / split / "samples.csv").read_bytes()).hexdigest()
    path = cache_dir / f"{split}_{weight_hash[:16]}_{manifest_hash[:16]}.npy"
    shape = (len(rows), 3, CHALLENGE_SAMPLES)
    if not path.is_file():
        enhance = FastEnhancer(weights)
        cache_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=cache_dir) as temp:
            temporary_path = Path(temp) / path.name
            waves = np.lib.format.open_memmap(
                temporary_path, mode="w+", dtype=np.float32, shape=shape
            )
            for index, row in enumerate(rows):
                noisy, clean = read_pair(root, row)
                waves[index] = noisy, enhance(noisy), clean
                if (index + 1) % 200 == 0:
                    print(f"{split} first pass {index + 1}/{len(rows)}", flush=True)
            del waves
            temporary_path.replace(path)
    waves = np.load(path, mmap_mode="r")
    if waves.shape != shape or waves.dtype != np.float32:
        raise ValueError(f"Invalid waveforms in {path}. Use a fresh --cache-dir.")
    return waves


def prepare_spectra(model, waves):
    # Cache the expensive STFT once and keep complex spectra in CPU memory.
    n_frames = waves.shape[-1] // model.hop + 1
    shape = (len(waves), 3, model.n_fft // 2 + 1, n_frames)
    spectra = torch.empty(shape, dtype=torch.complex64)
    with torch.no_grad():
        for start in range(0, len(waves), 32):
            batch = torch.from_numpy(waves[start:start + 32].copy()).flatten(0, 1)
            spectra[start:start + len(batch) // 3] = model.spectrum(batch).reshape(-1, *shape[1:])
    return spectra


@torch.inference_mode()
def validation(model, spectra, clean, batch_size, device):
    model.eval()
    totals = []
    for start in range(0, len(spectra), batch_size):
        inputs = spectra[start:start + batch_size].to(device)
        restored = model.restore(model(inputs[:, 0], inputs[:, 1]), clean.shape[-1]).cpu().numpy()
        for reference, prediction in zip(clean[start:start + batch_size], restored):
            totals.append(score(reference, pcm16(prediction)))
    return np.mean(totals, axis=0)


def train(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training = load_data(args.data_root, "train", args.first_pass, args.cache_dir)
    valid = load_data(args.data_root, "validation", args.first_pass, args.cache_dir)
    model = SpectralRefiner(args.channels)
    train_spec = prepare_spectra(model, training)
    val_spec = prepare_spectra(model, valid)
    loader = DataLoader(TensorDataset(train_spec), batch_size=args.batch_size, shuffle=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5, min_lr=2e-5
    )
    best = -np.inf
    start = monotonic()
    history = args.output_dir / "history.csv"
    with history.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["epoch", "train_loss", "pesq_wb", "snr_db", "learning_rate"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                loss = model.loss(batch[:, 0], batch[:, 1], batch[:, 2])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                total_loss += loss.item() * len(batch)
            learning_rate = optimizer.param_groups[0]["lr"]
            if epoch % args.eval_interval == 0 or epoch == args.epochs:
                pesq_wb, snr_db = validation(model, val_spec, valid[:, 2], args.batch_size, device)
                if pesq_wb > best:
                    best = pesq_wb
                    path = args.output_dir / "refiner.pt"
                    temporary_path = path.with_suffix(".tmp")
                    torch.save(
                        {"state_dict": model.state_dict(), "channels": args.channels, "epoch": epoch},
                        temporary_path,
                    )
                    temporary_path.replace(path)
                scheduler.step(pesq_wb)
            else:
                pesq_wb, snr_db = float("nan"), float("nan")
            train_loss = total_loss / len(training)
            writer.writerow([epoch, train_loss, pesq_wb, snr_db, learning_rate])
            handle.flush()
            print(f"epoch {epoch}: loss={train_loss:.5f} PESQ={pesq_wb:.4f} "
                  f"SNR={snr_db:.2f} dB elapsed={monotonic() - start:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--first-pass", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".waveform_cache/refiner"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    if min(args.threads, args.channels, args.batch_size, args.epochs, args.eval_interval) < 1:
        parser.error("Threads, channels, batch size, epochs and evaluation interval must be positive.")
    if args.channels % 8 or args.lr <= 0:
        parser.error("Channels must be a multiple of eight, and the learning rate must be positive.")
    train(args)
