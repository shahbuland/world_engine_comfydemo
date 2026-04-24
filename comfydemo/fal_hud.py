"""
FAL-based HUD element generator.

generate_hud_element(descriptor) -> np.ndarray [H, W, 3] uint8 RGB
chroma_key(img_rgb)              -> np.ndarray [H, W, 4] uint8 RGBA
"""
import os
import uuid
import urllib.request

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

import numpy as np
import cv2
import fal_client

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "fal_images")
TRANSPARENT_DIR = os.path.join(os.path.dirname(__file__), "fal_images_transparent")

HUD_PREFIX = "Opaque solid HUD element centred on a green screen. "

FAL_SETTINGS = {
    "num_images": 1,
    "aspect_ratio": "1:1",
    "output_format": "png",
    "safety_tolerance": "4",
    "resolution": "0.5K",
    "limit_generations": True,
    "thinking_level": "minimal",
}


def generate_hud_element(descriptor: str) -> np.ndarray:
    """
    Call fal-ai/nano-banana-2 and return the result as [H, W, 3] uint8 RGB.
    Raw PNG is saved to fal_images/.
    """
    os.makedirs(IMAGES_DIR, exist_ok=True)

    result = fal_client.subscribe(
        "fal-ai/nano-banana-2",
        arguments={"prompt": HUD_PREFIX + descriptor, **FAL_SETTINGS},
    )

    url = result["images"][0]["url"]
    data = np.frombuffer(urllib.request.urlopen(url).read(), np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)

    out_path = os.path.join(IMAGES_DIR, f"{uuid.uuid4().hex[:8]}.png")
    cv2.imwrite(out_path, img_bgr)
    print(f"Saved raw image: {out_path}")

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def chroma_key(img_rgb: np.ndarray) -> np.ndarray:
    """
    Remove green screen background from [H, W, 3] uint8 RGB.
    Returns [H, W, 4] uint8 RGBA — green pixels become transparent.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    # Green hue range in OpenCV (0-180 scale): ~35-85 covers yellow-green to cyan-green
    mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))

    rgba = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2RGBA)
    rgba[mask > 0, 3] = 0
    return rgba


if __name__ == "__main__":
    os.makedirs(TRANSPARENT_DIR, exist_ok=True)

    descriptor = (
        "Circular radar-type map similar to a popular FPS game, showing a map layout, "
        "some indicators for teammates and enemies, and some icons for environmental artifacts."
    )

    print("Generating HUD element...")
    img_rgb = generate_hud_element(descriptor)

    print("Applying chroma key...")
    rgba = chroma_key(img_rgb)

    out_path = os.path.join(TRANSPARENT_DIR, "test_hud.png")
    cv2.imwrite(out_path, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    print(f"Saved transparent image: {out_path}")
