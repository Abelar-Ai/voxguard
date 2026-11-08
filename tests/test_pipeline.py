import copy

import joblib
import numpy as np
import pytest
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from voxguard.cli import predict_scores
from voxguard.data import validate_partitions
from voxguard.features import FeatureConfig, extract, load_audio
from voxguard.metrics import compute_metrics, equal_error_rate


def tone(path, frequency=440, sample_rate=16000, seconds=0.4, channels=1):
    time = np.arange(int(sample_rate * seconds)) / sample_rate
    audio = 0.2 * np.sin(2 * np.pi * frequency * time)
    if channels == 2:
        audio = np.stack([audio, audio * 0.5], axis=1)
    sf.write(path, audio, sample_rate, subtype="FLOAT")


@pytest.mark.parametrize("sample_rate,channels", [(16000, 1), (22050, 2), (8000, 1)])
def test_audio_features_support_resampling_and_stereo(tmp_path, sample_rate, channels):
    path = tmp_path / "tone.wav"
    tone(path, sample_rate=sample_rate, channels=channels)
    config = FeatureConfig()
    audio = load_audio(path, config)
    assert audio.ndim == 1
    assert abs(len(audio) - 6400) <= 1
    assert np.isclose(np.max(np.abs(audio)), 1.0)
    features = extract(path, config)
    assert features.shape == (80,)
    assert np.isfinite(features).all()
    np.testing.assert_array_equal(features, extract(path, config))


@pytest.mark.parametrize("seconds,frequency,expected", [(0.01, 440, "at least"), (0.4, 0, "silent")])
def test_invalid_short_and_silent_audio_rejected(tmp_path, seconds, frequency, expected):
    path = tmp_path / "invalid.wav"
    tone(path, seconds=seconds, frequency=frequency)
    with pytest.raises(ValueError, match=expected):
        extract(path)


def test_long_audio_is_cropped_consistently(tmp_path):
    path = tmp_path / "long.wav"
    tone(path, seconds=5.0)
    assert len(load_audio(path)) == 64000


@pytest.mark.parametrize("scores,expected", [([0.1, 0.2, 0.8, 0.9], 0.0),
                                            ([0.9, 0.8, 0.2, 0.1], 1.0),
                                            ([0.5, 0.5, 0.5, 0.5], 0.5)])
def test_eer_handles_perfect_inverted_and_constant_scores(scores, expected):
    assert equal_error_rate([0, 0, 1, 1], scores) == pytest.approx(expected)


def test_false_alarms_and_misses_have_correct_class_orientation():
    result = compute_metrics([0, 0, 1, 1], [1, 0, 1, 1], [0.8, 0.1, 0.9, 0.7])
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["genuine_false_alarm_rate"] == 0.5
    assert result["synthetic_miss_rate"] == 0.0
    assert result["balanced_accuracy"] == 0.75


def rows():
    return [{"split": split, "utterance_id": f"{split}-{label}", "speaker_id": split,
             "sha256": f"{split}-hash-{label}", "label": label}
            for split in ("train", "dev", "eval") for label in (0, 1)]


@pytest.mark.parametrize("field", ["utterance_id", "speaker_id", "sha256"])
def test_partition_overlap_is_rejected(field):
    records = copy.deepcopy(rows())
    records[2][field] = records[0][field]
    with pytest.raises(ValueError, match="leakage|Duplicate"):
        validate_partitions(records)


def test_valid_partitions_pass():
    validate_partitions(rows())


def test_single_class_partition_rejected():
    records = rows()
    records[1]["label"] = 0
    with pytest.raises(ValueError, match="both classes"):
        validate_partitions(records)


def test_serialized_model_preserves_predictions_and_scaler_uses_training_only(tmp_path):
    training = np.array([[0., 1.], [1., 0.], [2., 1.], [3., 0.]])
    labels = [0, 0, 1, 1]
    development = np.array([[100., 100.], [-100., -100.]])
    model = make_pipeline(StandardScaler(), LogisticRegression()).fit(training, labels)
    expected = model.predict(development)
    np.testing.assert_allclose(model.named_steps["standardscaler"].mean_, training.mean(axis=0))
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    np.testing.assert_array_equal(restored.predict(development), expected)
    np.testing.assert_allclose(predict_scores(restored, development), predict_scores(model, development))
