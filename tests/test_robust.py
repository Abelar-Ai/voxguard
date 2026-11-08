import numpy as np
import pytest

from voxguard.robust_features import add_white_noise, represent
from voxguard.robust_train import select_threshold
from voxguard.fresh_benchmark import assert_holdout_safe


@pytest.mark.parametrize("representation,dimensions", [("mfcc", 80), ("lfcc", 160), ("combined", 240)])
def test_feature_representations_are_finite(representation, dimensions):
    audio = np.sin(2 * np.pi * 440 * np.arange(8000) / 16000).astype(np.float32)
    features = represent(audio, representation)
    assert features.shape == (dimensions,)
    assert np.isfinite(features).all()


@pytest.mark.parametrize("snr", [10, 20, 35])
def test_noise_has_requested_snr(snr):
    audio = np.sin(np.arange(16000)).astype(np.float32)
    augmented = add_white_noise(audio, snr, np.random.default_rng(475))
    measured = 10 * np.log10(np.mean(audio ** 2) / np.mean((augmented - audio) ** 2))
    assert measured == pytest.approx(snr, abs=1e-3)


def test_threshold_uses_synthetic_score_orientation():
    scores = np.array([-1.0, -0.5, 0.7, 1.0])
    labels = np.array([0, 0, 1, 1])
    threshold = select_threshold(labels, scores, 0)
    np.testing.assert_array_equal(scores >= threshold, labels)


def test_previously_inspected_evaluation_is_rejected():
    rows = [{"split": "eval", "utterance_id": "old", "speaker_id": "new", "sha256": "hash"}]
    metadata = {"training_ids": [], "training_speakers": [], "training_sha256": []}
    with pytest.raises(ValueError, match="Previously inspected"):
        assert_holdout_safe(rows, metadata, rows)


def test_training_overlap_in_fresh_evaluation_is_rejected():
    rows = [{"split": "eval", "utterance_id": "new", "speaker_id": "train-speaker", "sha256": "new"}]
    metadata = {"training_ids": [], "training_speakers": ["train-speaker"], "training_sha256": []}
    with pytest.raises(ValueError, match="training"):
        assert_holdout_safe(rows, metadata, [])
