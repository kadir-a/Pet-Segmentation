import torch
from PIL import Image

from app.model import PetSegmentationModel


def test_resizes_and_scales_rgb(loads):
    model = PetSegmentationModel("unused")
    tensor = model.preprocess(Image.new("RGB", (37, 18), (0, 128, 255)))
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    expected = torch.tensor([0, 128 / 255, 1]).view(3, 1, 1)
    torch.testing.assert_close(tensor, expected.expand_as(tensor))
