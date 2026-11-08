"""LFCC and mixed-feature experiments, separate from the frozen MFCC baseline."""
import librosa
import numpy as np
from scipy.fft import dct

from .features import FeatureConfig, features_from_audio


def add_white_noise(audio, snr_db, rng):
    audio = np.asarray(audio, dtype=np.float32)
    power = float(np.mean(audio ** 2))
    if power <= 0:
        raise ValueError("Noise augmentation requires non-silent audio")
    noise = rng.normal(size=len(audio)).astype(np.float32)
    scale = np.sqrt(power / (10 ** (snr_db / 10) * np.mean(noise ** 2)))
    return audio + scale * noise


def lfcc_features(audio):
    """40 LFCCs: 128 triangular linear bands, 1024 FFT, 25 ms window.

    This is our LFCC implementation, not an exact reproduction of the official
    LFCC-GMM or LFCC-LCNN baseline. Each statistic uses available frames only.
    """
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-8 or not np.isfinite(audio).all():
        raise ValueError("Expected finite non-silent audio")
    spectrum = np.abs(librosa.stft(audio / peak, n_fft=1024, win_length=400,
                                   hop_length=160, center=False)) ** 2
    frequencies = np.linspace(0, 8000, spectrum.shape[0])
    edges = np.linspace(0, 8000, 130)
    left = (frequencies[None, :] - edges[:-2, None]) / (edges[1:-1, None] - edges[:-2, None])
    right = (edges[2:, None] - frequencies[None, :]) / (edges[2:, None] - edges[1:-1, None])
    filters = np.maximum(0, np.minimum(left, right))
    cepstra = dct(np.log(np.maximum(filters @ spectrum, 1e-10)), type=2, norm="ortho", axis=0)[:40]
    delta = librosa.feature.delta(cepstra, width=9, mode="nearest")
    vector = np.concatenate([cepstra.mean(axis=1), cepstra.std(axis=1), delta.mean(axis=1), delta.std(axis=1)])
    if not np.isfinite(vector).all():
        raise ValueError("Non-finite LFCC features")
    return vector.astype(np.float32)


def represent(audio, representation):
    if representation == "mfcc":
        return features_from_audio(audio, FeatureConfig())
    if representation == "lfcc":
        return lfcc_features(audio)
    if representation == "combined":
        return np.concatenate([features_from_audio(audio), lfcc_features(audio)])
    raise ValueError(f"Unknown representation {representation}")
