import json
from pathlib import Path

import numpy as np

from .fastenhancer import FastEnhancer


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "submission.json"


def load_enhancer(config_path, threads=2):
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    def locate(key):
        path = config_path.parent / config[key]
        if not path.is_file():
            raise FileNotFoundError(f"Missing local model: {path}")
        return path

    if "mpsenet" in config:
        from .mpsenet.enhance import MPSENet

        primary = MPSENet(locate("mpsenet"), locate("mpsenet_settings"), threads)
        if "fastenhancer" not in config:
            return primary

    secondary = FastEnhancer(locate("fastenhancer"), threads)
    if "refiner" in config:
        secondary = refine(secondary, locate("refiner"), config["refinement_strength"], threads)
    if "mpsenet" not in config:
        return secondary

    mix = float(config["fastenhancer_mix"])
    if not 0 <= mix <= 1:
        raise ValueError("fastenhancer_mix must be between zero and one.")

    def enhance(waveform):
        main = primary(waveform)
        return main + mix * (secondary(waveform) - main)

    return enhance


def refine(first_pass, checkpoint_path, strength, threads):

    import torch

    from .refinement import SpectralRefiner

    torch.set_num_threads(threads)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = SpectralRefiner(checkpoint["channels"]).eval()
    model.load_state_dict(checkpoint["state_dict"])

    @torch.inference_mode()
    def enhance(waveform):
        restored = first_pass(waveform)
        noisy = model.spectrum(torch.from_numpy(np.ascontiguousarray(waveform))[None])
        initial = model.spectrum(torch.from_numpy(restored)[None])
        estimate = model(noisy, initial)
        if strength != 1.0:
            estimate = initial + strength * (estimate - initial)
        return model.restore(estimate, len(waveform))[0].numpy()

    return enhance
