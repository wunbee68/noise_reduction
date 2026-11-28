import argparse
import csv
from pathlib import Path

import numpy as np

from denoise.audio import SAMPLE_RATE, read_wav, wav_files, write_wav
from denoise.data import manifest, read_pair
from denoise.metrics import pcm16, score
from denoise.models import DEFAULT_CONFIG, load_enhancer


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_examples(directory, results, waves):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import stft

    directory.mkdir(parents=True, exist_ok=True)
    ranked = sorted(range(len(results)), key=lambda i: results[i]["delta_pesq_wb"])
    for label, index in (("lowest", ranked[0]), ("median", ranked[len(ranked) // 2]),
                         ("highest", ranked[-1])):
        row = results[index]
        folder = directory / f'{label}_{row["segment_id"]}'
        folder.mkdir(exist_ok=True)
        clips = waves[index]
        spectra = []
        for name, clip in zip(("noisy", "enhanced", "clean"), clips):
            write_wav(folder / f"{name}.wav", clip[:, None], SAMPLE_RATE)
            hz, time, spectrum = stft(clip, fs=SAMPLE_RATE, nperseg=512, noverlap=384)
            spectra.append(20 * np.log10(np.maximum(np.abs(spectrum), 1e-8)))

        peak = max(float(spectrum.max()) for spectrum in spectra)
        figure, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(8, 7),
                                    layout="constrained")
        for axis, title, spectrum in zip(axes, ("Noisy", "Enhanced", "Clean"), spectra):
            image = axis.pcolormesh(time, hz / 1000, spectrum, shading="auto",
                                    vmin=peak - 75, vmax=peak, cmap="magma")
            axis.set(ylabel="Frequency (kHz)", title=title)
        axes[-1].set(xlabel="Time (s)")
        figure.colorbar(image, ax=axes, label="Magnitude (dB)")
        figure.suptitle(
            f'{row["noise_category"]} · PESQ-WB {row["noisy_pesq_wb"]:.2f} → '
            f'{row["enhanced_pesq_wb"]:.2f}'
        )
        figure.savefig(folder / "spectrogram.png", dpi=180)
        plt.close(figure)


def evaluate(args):
    data_rows = manifest(args.data_root, "validation")
    if args.predictions_dir is not None:
        directory = args.predictions_dir
        expected = {directory / Path(row["noisy_path"]).name for row in data_rows}
        if len(expected) != len(data_rows) or set(wav_files(directory)) != expected:
            raise ValueError("Output WAV names must match the entire Validation manifest.")
    else:
        enhance = load_enhancer(args.config, args.threads)

    results, waves = [], []
    for index, row in enumerate(data_rows, 1):
        noisy, clean = read_pair(args.data_root, row)
        if args.predictions_dir is None:
            restored = enhance(noisy)
            if restored.shape != noisy.shape or not np.isfinite(restored).all():
                raise ValueError(f'Invalid output for {row["segment_id"]}')
            prediction = pcm16(restored)
        else:
            path = directory / Path(row["noisy_path"]).name
            prediction, _ = read_wav(path, challenge_format=True)
            prediction = prediction[:, 0]
        try:
            noisy_pesq, noisy_snr = score(clean, noisy)
            output_pesq, output_snr = score(clean, prediction)
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError(f'Metric computation failed for {row["segment_id"]}') from exc
        results.append({
            "segment_id": row["segment_id"],
            "noise_category": row["noise_category"],
            "target_snr_db": row["target_snr_db"],
            "noisy_pesq_wb": noisy_pesq,
            "enhanced_pesq_wb": output_pesq,
            "delta_pesq_wb": output_pesq - noisy_pesq,
            "noisy_snr_db": noisy_snr,
            "enhanced_snr_db": output_snr,
            "delta_snr_db": output_snr - noisy_snr,
        })
        if args.examples_dir is not None:
            waves.append((noisy, prediction, clean))
        if index % 50 == 0 or index == len(data_rows):
            print(f"{index}/{len(data_rows)} Validation WAVs scored", flush=True)

    summary = []
    for condition in ("noisy", "enhanced"):
        summary.append({
            "condition": condition,
            "files": len(results),
            "mean_pesq_wb": np.mean([row[f"{condition}_pesq_wb"] for row in results]),
            "mean_snr_db": np.mean([row[f"{condition}_snr_db"] for row in results]),
        })
    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.results_dir / "per_file.csv", results)
    write_csv(args.results_dir / "summary.csv", summary)
    if args.examples_dir is not None:
        save_examples(args.examples_dir, results, waves)
    for row in summary:
        print(f'{row["condition"]:8} PESQ-WB {row["mean_pesq_wb"]:.4f}  '
              f'SNR {row["mean_snr_db"]:.4f} dB')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score all Validation pairs with equal file weights.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--predictions-dir", type=Path)
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/submission_validation"))
    parser.add_argument("--examples-dir", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    evaluate(parser.parse_args())
