from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


SAMPLE_RATE = 16_000
CHALLENGE_SAMPLES = 48_000


def read_wav(path, *, challenge_format=False):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Empty or non-finite audio: {path}")
    if challenge_format and (rate != SAMPLE_RATE or audio.shape != (CHALLENGE_SAMPLES, 1)):
        raise ValueError(f"Expected 16 kHz, mono, 48,000 samples: {path}")
    return audio, rate


def enhance_audio(audio, rate, enhance):
    """Apply the 16 kHz mono model independently to each channel."""
    if rate != SAMPLE_RATE:
        model_input = resample_poly(audio, SAMPLE_RATE, rate, axis=0)
    else:
        model_input = audio
    restored = np.stack([enhance(channel) for channel in model_input.T], axis=1)
    if restored.shape != model_input.shape or not np.isfinite(restored).all():
        raise ValueError("The model returned invalid samples or changed the waveform shape.")
    if rate != SAMPLE_RATE:
        restored = resample_poly(restored, rate, SAMPLE_RATE, axis=0)[:len(audio)]
    if restored.shape != audio.shape:
        raise ValueError("Resampling changed the output shape.")
    return restored


def write_wav(path, audio, rate):
    sf.write(path, audio, rate, format="WAV", subtype="PCM_16")
    info = sf.info(path)
    if (info.samplerate, info.frames, info.channels) != (rate, *audio.shape):
        raise RuntimeError(f"Output WAV format verification failed: {path}")


def wav_files(directory: Path):
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    )
