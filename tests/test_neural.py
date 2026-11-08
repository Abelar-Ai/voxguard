import numpy as np
import pytest

torch = pytest.importorskip("torch")
from voxguard.neural import SpectrogramCNN, spectrogram


def test_cnn_supports_variable_lengths_and_backward_pass():
    torch.set_num_threads(1)
    model = SpectrogramCNN()
    values = torch.randn(2, 1, 64, 400)
    logits = model(values, torch.tensor([7, 397]))
    assert logits.shape == (2,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_spectrogram_has_explicit_valid_length():
    audio = np.sin(np.arange(16000)).astype(np.float32)
    values, frames = spectrogram(audio)
    assert values.shape == (64, 400)
    assert 0 < frames < 400
    assert np.isfinite(values).all()
    assert np.all(values[:, frames:] == -80)
