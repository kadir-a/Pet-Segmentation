from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


def upload(client, fmt):
    stream = BytesIO()
    Image.new("RGB", (38, 17), "red").save(stream, format=fmt)
    return client.post("/predict", files={"file": (f"pet.{fmt}", stream.getvalue(),
                                                   f"image/{fmt.lower()}")})


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_mask_pixels(loads, fmt):
    with TestClient(app) as client:
        response = upload(client, fmt)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    with Image.open(BytesIO(response.content)) as mask:
        assert (mask.format, mask.mode, mask.size) == ("PNG", "L", (224, 224))
        pixels = np.asarray(mask)
        assert (pixels[:, :112] == 0).all()
        assert (pixels[:, 112:] == 255).all()


def test_one_load_per_startup(loads):
    assert len(loads) == 0
    with TestClient(app) as client:
        assert len(loads) == 1
        for fmt in ["JPEG", "PNG"]:
            assert upload(client, fmt).status_code == 200
        assert len(loads) == 1


@pytest.mark.parametrize("name, data, mime", [
    ("bad.png", b"corrupt", "image/png"), ("empty.png", b"", "image/png"),
    ("readme.txt", b"text", "text/plain"),
])
def test_invalid_upload(loads, name, data, mime):
    with TestClient(app) as client:
        response = client.post("/predict", files={"file": (name, data, mime)})
    assert response.status_code == 400
    assert response.json()["detail"]
