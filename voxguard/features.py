"""Shared preprocessing for training and inference."""
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 16000
    max_seconds: float = 4.0
    min_seconds: float = 0.1
    n_mfcc: int = 20
    n_mels: int = 64
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160


def load_audio(path, config=FeatureConfig()):
    info = sf.info(str(path))
    if info.frames / info.samplerate < config.min_seconds:
        raise ValueError(f"Audio must contain at least {config.min_seconds} seconds")
    audio, sr = sf.read(str(path), frames=int(info.samplerate * config.max_seconds),
                        dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains non-finite samples")
    if sr != config.sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=config.sample_rate)
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-8:
        raise ValueError("Audio is silent")
    return audio / peak


def features_from_audio(audio, config=FeatureConfig()):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or len(audio) < config.n_fft:
        raise ValueError("Expected a one-dimensional audio signal of at least n_fft samples")
    if not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-8:
        raise ValueError("Expected finite, non-silent audio")
    audio = audio / np.max(np.abs(audio))
    mfcc = librosa.feature.mfcc(y=audio, sr=config.sample_rate, n_mfcc=config.n_mfcc,
                               n_mels=config.n_mels, n_fft=config.n_fft,
                               win_length=config.win_length, hop_length=config.hop_length,
                               center=False)
    delta = librosa.feature.delta(mfcc, width=9, mode="nearest")
    features = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1),
                               delta.mean(axis=1), delta.std(axis=1)]).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Feature extraction produced non-finite values")
    return features


def extract(path, config=FeatureConfig()):
    return features_from_audio(load_audio(Path(path), config), config)
