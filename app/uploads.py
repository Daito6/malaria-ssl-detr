"""Safe decoding of uploaded images, in memory only (nothing is written to disk).

Checks, in order: size limit, JPEG / PNG signature, pixel count from the header (before decoding, so a small file
that decodes to a huge image is rejected cheaply), full decode. Decoding uses cv2.imdecode(IMREAD_COLOR) as
malaria_ssl.data.detection.read_rgb, so an uploaded file gives the same RGB array as the evaluation pipeline.
"""

from __future__ import annotations

import io
import warnings

import cv2
import numpy as np
from PIL import Image

MAX_BYTES = 25 * 1024 * 1024
MAX_PIXELS = 50_000_000
MIN_SIDE = 32
SIGNATURES = {b"\xff\xd8\xff": "jpeg", b"\x89PNG\r\n\x1a\n": "png"}


class UploadError(ValueError):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def sniff_format(data: bytes) -> str | None:
    return next((fmt for sig, fmt in SIGNATURES.items() if data.startswith(sig)), None)


def decode_image(data: bytes, max_bytes: int = MAX_BYTES, max_pixels: int = MAX_PIXELS) -> np.ndarray:
    """Uploaded bytes -> RGB uint8 (H, W, 3) or UploadError."""
    if not data:
        raise UploadError("empty_file", "the uploaded file is empty")
    if len(data) > max_bytes:
        raise UploadError("file_too_large", f"file is {len(data)} bytes, the limit is {max_bytes}", 413)
    fmt = sniff_format(data)
    if fmt is None:
        raise UploadError("unsupported_format", "only JPEG and PNG images are accepted", 415)
    try:
        with warnings.catch_warnings():  # PIL warns above its own 89 MP default; our limit is checked below
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as im:  # header only
                w, h = im.size
    except Image.DecompressionBombError as err:
        raise UploadError("image_too_large", f"image exceeds the pixel limit of {max_pixels}: {err}", 413) from err
    except Exception as err:  # noqa: BLE001 - any parser failure means a broken file
        raise UploadError("invalid_image", f"cannot read the {fmt.upper()} header: {err}") from err
    if w * h > max_pixels:
        raise UploadError("image_too_large", f"image is {w}x{h} = {w * h} pixels, the limit is {max_pixels}", 413)
    if min(w, h) < MIN_SIDE:
        raise UploadError("image_too_small", f"image is {w}x{h}, the minimum side is {MIN_SIDE} px")
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise UploadError("invalid_image", f"the {fmt.upper()} data cannot be decoded")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
