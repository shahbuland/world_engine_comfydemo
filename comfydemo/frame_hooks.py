"""
Frame hooks: transform the current frame batch while the engine is paused.

To add a new effect, subclass FrameHook and override process().
The hook runs in a background thread so the pygame window stays responsive.
"""
import io
import numpy as np


class FrameHook:
    """Passthrough base class. Override process() to transform frames."""

    def process(self, frames: np.ndarray) -> np.ndarray:
        """
        frames: [4, H, W, 3] uint8
        Returns: [4, H, W, 3] uint8 — same shape, modified content
        """
        return frames


class HTTPImg2ImgHook(FrameHook):
    """
    Generic image2image hook. POSTs the last frame as a PNG to any HTTP
    endpoint and expects a PNG image back. Tiles the result across all 4 frames.

    Wire this up to any img2img API (AUTOMATIC1111 /sdapi/v1/img2img,
    Stability AI, a local script, whatever).

    The endpoint receives:
        POST <url>
        Content-Type: multipart/form-data
        Field: "image" = PNG file

    And should return a raw PNG image in the response body.
    """

    def __init__(self, url: str, timeout: float = 30.0, extra_fields: dict = None):
        self.url = url
        self.timeout = timeout
        self.extra_fields = extra_fields or {}

    def process(self, frames: np.ndarray) -> np.ndarray:
        import requests
        from PIL import Image

        last_frame = frames[-1]  # [H, W, 3]
        h, w = last_frame.shape[:2]

        buf = io.BytesIO()
        Image.fromarray(last_frame).save(buf, format="PNG")
        buf.seek(0)

        files = {"image": ("frame.png", buf, "image/png")}
        resp = requests.post(self.url, files=files, data=self.extra_fields, timeout=self.timeout)
        resp.raise_for_status()

        result = np.array(Image.open(io.BytesIO(resp.content)).convert("RGB").resize((w, h)))
        return np.stack([result] * 4).astype(np.uint8)
