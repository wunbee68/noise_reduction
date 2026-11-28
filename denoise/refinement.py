import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.Conv1d(channels, channels, 5, padding=2 * dilation, dilation=dilation,
                      groups=channels),
            nn.Conv1d(channels, channels, 1),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, features):
        return features + self.layers(features)


class SpectralRefiner(nn.Module):
    n_fft = 512
    hop = 256
    compression = 0.3

    def __init__(self, channels=96):
        super().__init__()
        bins = self.n_fft // 2 + 1
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)
        self.input = nn.Conv1d(3 * bins, channels, 1)
        self.blocks = nn.Sequential(*(ResidualBlock(channels, rate) for rate in (1, 2, 4, 8, 16, 1)))
        self.output = nn.Conv1d(channels, 3 * bins, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def spectrum(self, wave):
        return torch.stft(
            wave, self.n_fft, self.hop, window=self.window,
            pad_mode="constant", return_complex=True,
        )

    def restore(self, spectrum, length):
        return torch.istft(spectrum, self.n_fft, self.hop, window=self.window, length=length)

    def forward(self, noisy, first_pass):
        noisy_mag, restored_mag = noisy.abs(), first_pass.abs()
        noise_mag = (noisy - first_pass).abs()
        features = torch.cat(
            [torch.log1p(10 * magnitude) for magnitude in (noisy_mag, restored_mag, noise_mag)], dim=1
        )
        hidden = self.blocks(self.input(features))
        gain, phase, add_back = self.output(hidden).tanh().chunk(3, dim=1)
        # Learn conservative magnitude/phase corrections plus a bounded noisy residual.
        corrected = first_pass * torch.polar((1.6 * gain).exp(), 0.35 * phase)
        return corrected + 0.2 * add_back * (noisy - first_pass)

    def loss(self, noisy, first_pass, clean):
        estimate = self(noisy, first_pass)
        estimated_mag = estimate.abs().clamp_min(1e-7)
        clean_mag = clean.abs().clamp_min(1e-7)
        estimated_comp = estimated_mag.pow(self.compression)
        clean_comp = clean_mag.pow(self.compression)
        estimated_complex = estimate * (estimated_comp / estimated_mag)
        clean_complex = clean * (clean_comp / clean_mag)
        magnitude_loss = (estimated_comp - clean_comp).abs().mean()
        phase_loss = (estimated_complex - clean_complex).abs().mean()
        return magnitude_loss + 0.5 * phase_loss
