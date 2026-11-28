import numpy as np

from .audio import SAMPLE_RATE


def score(clean, prediction):
    from pesq import pesq

    quality = float(pesq(SAMPLE_RATE, clean, prediction, mode="wb"))
    if not np.isfinite(quality):
        raise ValueError(f"PESQ-WB failed with score {quality}")
    clean = clean.astype(np.float64)
    error = clean - prediction
    signal_energy = np.dot(clean, clean)
    error_energy = np.dot(error, error)
    if signal_energy == 0:
        raise ValueError("SNR is undefined for a silent reference")
    snr = float(10 * np.log10(signal_energy / error_energy)) if error_energy else float("inf")
    return quality, snr


def pcm16(audio):
    # Match libsndfile's float-to-PCM16 writing before computing Validation metrics.
    return np.floor(np.clip(audio, -1, 32767 / 32768) * 32768) / 32768
