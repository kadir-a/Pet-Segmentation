import pytest
import torch

from app.model import PetSegmentationModel


class SplitMask(torch.nn.Module):
    def forward(self, images):
        logits = images.new_zeros((images.shape[0], 2, 224, 224))
        logits[:, 1, :, 112:] = 2
        return logits


@pytest.fixture
def loads(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def load(self, path):
        calls.append(path)
        return SplitMask().eval()

    monkeypatch.setattr(PetSegmentationModel, "_load_model", load)
    return calls
