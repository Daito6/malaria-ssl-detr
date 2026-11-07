"""Upload decoding: limits, formats, broken files, parity with the evaluation's read_rgb."""

from __future__ import annotations

import io
import struct
import zlib

import cv2
import numpy as np
import pytest
from PIL import Image

from app.uploads import UploadError, decode_image
from malaria_ssl.data.detection import read_rgb


def _png(w: int, h: int, color=(200, 120, 160)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def _png_header_only(w: int, h: int) -> bytes:
    """A PNG whose header claims w x h (decompression-bomb shape) with almost no pixel data."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"\0")) + chunk(b"IEND", b"")


def _code(data: bytes, **kw) -> str:
    with pytest.raises(UploadError) as e:
        decode_image(data, **kw)
    return e.value.code


def test_valid_png_matches_read_rgb(val_image_path):
    img = decode_image(val_image_path.read_bytes())
    np.testing.assert_array_equal(img, read_rgb(val_image_path))


def test_valid_jpeg():
    ok, enc = cv2.imencode(".jpg", np.full((300, 400, 3), 128, np.uint8))
    assert ok
    img = decode_image(enc.tobytes())
    assert img.shape == (300, 400, 3) and img.dtype == np.uint8


def test_empty():
    assert _code(b"") == "empty_file"


@pytest.mark.parametrize("data", [b"GIF89a" + b"\0" * 100, b"hello world", b"%PDF-1.4 ..."])
def test_unsupported_format(data):
    assert _code(data) == "unsupported_format"


def test_truncated_png():
    assert _code(_png(400, 300)[:60]) == "invalid_image"


@pytest.mark.parametrize("side", [10_000, 20_000])  # 100 MP: over our limit; 400 MP: PIL's own bomb guard
def test_too_many_pixels_rejected_from_header(side):
    assert _code(_png_header_only(side, side)) == "image_too_large"


def test_file_too_large():
    assert _code(_png(400, 300), max_bytes=100) == "file_too_large"


def test_too_small():
    assert _code(_png(10, 10)) == "image_too_small"


def test_large_image_within_limits():
    img = decode_image(_png(6000, 4500))  # 27 MP, under the 50 MP limit
    assert img.shape == (4500, 6000, 3)
