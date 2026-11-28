import json
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from .model import MPNet


class MPSENet:
    def __init__(self, weights, settings, threads=2):
        torch.set_num_threads(threads)
        self.settings = SimpleNamespace(**json.loads(settings.read_text(encoding="utf-8")))
        self.window = torch.hann_window(self.settings.win_size)
        self.model = MPNet(self.settings, num_tsblocks=self.settings.num_tsblocks).eval()
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["generator"])

    @torch.inference_mode()
    def __call__(self, waveform):
        # The upstream checkpoint uses unit RMS and one clip per forward pass.
        wave = torch.from_numpy(np.ascontiguousarray(waveform)).unsqueeze(0)
        length = wave.shape[-1]
        if not torch.any(wave):
            return np.zeros_like(waveform)
        settings = self.settings
        if length <= settings.n_fft // 2:
            wave = F.pad(wave, (0, settings.n_fft // 2 + 1 - length))
        gain = wave.square().mean().clamp_min(1e-12).rsqrt()
        spectrum = torch.stft(
            wave * gain, settings.n_fft, settings.hop_size, win_length=settings.win_size,
            window=self.window, return_complex=True,
        )
        magnitude = (spectrum.real.square() + spectrum.imag.square() + 1e-9).sqrt()
        magnitude = magnitude.pow(settings.compress_factor)
        phase = torch.atan2(spectrum.imag + 1e-10, spectrum.real + 1e-5)
        magnitude, phase, _ = self.model(magnitude, phase)
        restored = torch.polar(magnitude.pow(1 / settings.compress_factor), phase)
        result = torch.istft(
            restored, settings.n_fft, settings.hop_size, win_length=settings.win_size,
            window=self.window, length=wave.shape[-1],
        ) / gain
        return result[0, :length].numpy()
