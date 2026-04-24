"""
Waypoint-1.5 demo with AI HUD editor.

Controls (playing):
  WASD / mouse / 1-4 / arrows   game input
  P                              pause / resume
  6                              enter HUD edit mode
  7                              reset scene
  8                              random new seed
  Escape                         quit

Controls (edit mode):
  Enter                          open prompt dialogue
  6                              lock everything, restart world from composited frame
  (while positioning an element)
  WASD                           move element
  Up / Down arrows               scale up / down
  Space                          lock element in place
  Enter                          lock element + open new prompt

Controls (dialogue box):
  Type                           describe the HUD element
  Enter                          generate
  Escape                         cancel
"""
import sys
import os
import threading
import argparse
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

import numpy as np
import cv2
import pygame

sys.path.insert(0, os.path.dirname(__file__))

from engine_loop import EngineLoop
from play import (
    KEY_MAP, RESOLUTION_360P, DISPLAY_RESOLUTION, MOUSE_SENSITIVITY,
    MOUSE_LEFT_BTN, MOUSE_RIGHT_BTN, MOUSE_MIDDLE_BTN,
    load_seed_frame, blit_frame,
)
from fal_hud import generate_hud_element, chroma_key
from world_engine import CtrlInput

# ---------------------------------------------------------------------------
# Edit mode constants
# ---------------------------------------------------------------------------
MOVE_SPEED  = 3     # pixels per frame while key held
SCALE_SPEED = 0.008 # scale fraction per frame while key held
INITIAL_SCALE = 0.4 # element height as fraction of frame height


# ---------------------------------------------------------------------------
# HUD element
# ---------------------------------------------------------------------------
@dataclass
class HUDElement:
    img_rgba: np.ndarray   # [H, W, 4] RGBA uint8
    x: float               # center x on RESOLUTION_360P frame
    y: float               # center y on RESOLUTION_360P frame
    scale: float = INITIAL_SCALE


def _composite(frame: np.ndarray, elem: HUDElement) -> None:
    """Alpha-composite elem onto frame in-place. frame is [H, W, 3] uint8 RGB."""
    fh, fw = frame.shape[:2]
    eh, ew = elem.img_rgba.shape[:2]

    target_h = max(1, int(fh * elem.scale))
    target_w = max(1, int(ew * target_h / eh))
    scaled = cv2.resize(elem.img_rgba, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    x1 = int(elem.x - target_w / 2)
    y1 = int(elem.y - target_h / 2)

    # Source and destination slices (clamped to frame bounds)
    sx1 = max(0, -x1);        sy1 = max(0, -y1)
    dx1 = max(0,  x1);        dy1 = max(0,  y1)
    dx2 = min(fw, x1 + target_w); dy2 = min(fh, y1 + target_h)
    sx2 = sx1 + (dx2 - dx1);  sy2 = sy1 + (dy2 - dy1)

    if dx2 <= dx1 or dy2 <= dy1:
        return

    alpha = scaled[sy1:sy2, sx1:sx2, 3:4].astype(np.float32) / 255.0
    fg    = scaled[sy1:sy2, sx1:sx2, :3].astype(np.float32)
    bg    = frame[dy1:dy2, dx1:dx2].astype(np.float32)
    frame[dy1:dy2, dx1:dx2] = (alpha * fg + (1.0 - alpha) * bg).astype(np.uint8)


def composite_all(frame: np.ndarray, elements: list) -> np.ndarray:
    """Return a new frame with all HUDElements composited on top."""
    result = frame.copy()
    for elem in elements:
        _composite(result, elem)
    return result


# ---------------------------------------------------------------------------
# UI drawing
# ---------------------------------------------------------------------------
def draw_dialogue(screen: pygame.Surface, font: pygame.font.Font,
                  typed_text: str, generating: bool) -> None:
    W, H = DISPLAY_RESOLUTION
    bw, bh = W - 80, 120
    bx, by = 40, H // 2 - bh // 2

    overlay = pygame.Surface((bw, bh), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 210))
    screen.blit(overlay, (bx, by))

    if generating:
        label = font.render("Generating... please wait", True, (100, 200, 255))
        screen.blit(label, (bx + 20, by + 45))
    else:
        screen.blit(font.render("Describe HUD element:", True, (200, 200, 200)), (bx + 20, by + 15))
        cursor = "|" if pygame.time.get_ticks() % 1000 < 500 else " "
        screen.blit(font.render(typed_text + cursor, True, (255, 255, 255)), (bx + 20, by + 50))
        screen.blit(font.render("Enter: generate   Esc: cancel", True, (130, 130, 130)), (bx + 20, by + 90))


def draw_edit_hud(screen: pygame.Surface, font: pygame.font.Font, positioning: bool) -> None:
    if positioning:
        text = "WASD: move  |  ↑↓: scale  |  Space: lock  |  Enter: lock+new  |  6: done"
    else:
        text = "EDIT MODE  |  Enter: add element  |  6: done & restart world"
    screen.blit(font.render(text, True, (255, 220, 0)), (10, 10))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Waypoint-1.5 + HUD editor")
    parser.add_argument("--model", default="Overworld/Waypoint-1.5-1B-360P")
    parser.add_argument("--prompt", default=None)
    args = parser.parse_args()

    loop = EngineLoop(args.model)
    if args.prompt:
        loop.set_prompt(args.prompt)

    seed = load_seed_frame(RESOLUTION_360P)
    seed_x4 = np.stack([seed] * 4)

    pygame.init()
    screen = pygame.display.set_mode(DISPLAY_RESOLUTION)
    pygame.display.set_caption("Waypoint-1.5 + HUD Editor")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)

    blit_frame(screen, seed)
    pygame.display.flip()
    loop.start(seed_x4)

    # Game state
    current_frames = seed_x4
    pressed: set[int] = set()
    scroll = 0

    # Edit mode state
    # states: "playing" | "edit_idle" | "edit_typing" | "edit_generating" | "edit_positioning"
    state = "playing"
    hud_elements: list[HUDElement] = []   # locked elements
    active_elem: HUDElement | None = None  # element currently being positioned
    typed_text = ""

    def start_generation(text: str):
        nonlocal state, active_elem
        state = "edit_generating"

        def _gen():
            nonlocal state, active_elem
            try:
                img_rgb = generate_hud_element(text)
                rgba = chroma_key(img_rgb)
                cx = RESOLUTION_360P[0] / 2
                cy = RESOLUTION_360P[1] / 2
                active_elem = HUDElement(img_rgba=rgba, x=cx, y=cy)
                state = "edit_positioning"
            except Exception as e:
                print(f"Generation error: {e}")
                state = "edit_idle"

        threading.Thread(target=_gen, daemon=True).start()

    def lock_active():
        nonlocal active_elem
        if active_elem is not None:
            hud_elements.append(active_elem)
            active_elem = None

    def exit_edit_mode():
        nonlocal state, current_frames
        lock_active()
        # Composite all HUD elements onto the last frame
        base = current_frames[-1]
        composited = composite_all(base, hud_elements)
        # Repeat 4x for temporal compression, push through VAE, restart world
        frames_x4 = np.stack([composited] * 4).astype(np.uint8)
        loop.restart_from(frames_x4)
        # Use the VAE-decoded version so the compression artefacts are visible
        # and we know the frame genuinely went through the model
        current_frames = loop.last_frames
        # Clear elements — they're baked into the world now, don't draw as overlay
        hud_elements.clear()
        loop.resume()
        state = "playing"

    running = True
    while running:
        scroll = 0
        mouse_dx = mouse_dy = 0.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:

                if state == "edit_typing":
                    if event.key == pygame.K_RETURN and typed_text.strip():
                        start_generation(typed_text.strip())
                        typed_text = ""
                    elif event.key == pygame.K_ESCAPE:
                        state = "edit_idle"
                        typed_text = ""
                    elif event.key == pygame.K_BACKSPACE:
                        typed_text = typed_text[:-1]
                    elif event.unicode and event.unicode.isprintable():
                        typed_text += event.unicode

                elif state in ("edit_idle", "edit_positioning", "edit_generating"):
                    if event.key == pygame.K_6:
                        exit_edit_mode()
                    elif event.key == pygame.K_RETURN and state != "edit_generating":
                        lock_active()
                        state = "edit_typing"
                        typed_text = ""
                    elif event.key == pygame.K_SPACE and state == "edit_positioning":
                        lock_active()
                        state = "edit_idle"

                elif state == "playing":
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_p:
                        loop.resume() if loop.is_paused else loop.pause()
                    elif event.key == pygame.K_6:
                        loop.pause()
                        state = "edit_idle"
                    elif event.key == pygame.K_7:
                        loop.reset_scene()
                    elif event.key == pygame.K_8:
                        def _reseed():
                            try:
                                loop.reseed(np.stack([load_seed_frame(RESOLUTION_360P)] * 4))
                            except Exception as e:
                                print(f"Reseed error: {e}")
                        threading.Thread(target=_reseed, daemon=True).start()

                    owl = KEY_MAP.get(event.key)
                    if owl is not None:
                        pressed.add(owl)

            elif event.type == pygame.KEYUP:
                if state == "playing":
                    owl = KEY_MAP.get(event.key)
                    if owl is not None:
                        pressed.discard(owl)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: pressed.add(MOUSE_LEFT_BTN)
                elif event.button == 3: pressed.add(MOUSE_RIGHT_BTN)
                elif event.button == 2: pressed.add(MOUSE_MIDDLE_BTN)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1: pressed.discard(MOUSE_LEFT_BTN)
                elif event.button == 3: pressed.discard(MOUSE_RIGHT_BTN)
                elif event.button == 2: pressed.discard(MOUSE_MIDDLE_BTN)
            elif event.type == pygame.MOUSEMOTION:
                mouse_dx += event.rel[0] * MOUSE_SENSITIVITY
                mouse_dy += event.rel[1] * MOUSE_SENSITIVITY
            elif event.type == pygame.MOUSEWHEEL:
                scroll = 1 if event.y > 0 else -1

        # Continuous WASD / arrow movement for active element
        if state == "edit_positioning" and active_elem is not None:
            k = pygame.key.get_pressed()
            if k[pygame.K_w]: active_elem.y -= MOVE_SPEED
            if k[pygame.K_s]: active_elem.y += MOVE_SPEED
            if k[pygame.K_a]: active_elem.x -= MOVE_SPEED
            if k[pygame.K_d]: active_elem.x += MOVE_SPEED
            if k[pygame.K_UP]:
                active_elem.scale = min(2.0, active_elem.scale + SCALE_SPEED)
            if k[pygame.K_DOWN]:
                active_elem.scale = max(0.05, active_elem.scale - SCALE_SPEED)
            active_elem.x = max(0, min(RESOLUTION_360P[0], active_elem.x))
            active_elem.y = max(0, min(RESOLUTION_360P[1], active_elem.y))

        # Step engine (only in playing state)
        if state == "playing":
            ctrl = CtrlInput(button=set(pressed), mouse=[mouse_dx, mouse_dy], scroll_wheel=scroll)
            current_frames = loop.step(ctrl)

        # Render
        all_elems = hud_elements + ([active_elem] if active_elem is not None else [])
        display_frame = composite_all(current_frames[-1], all_elems) if all_elems else current_frames[-1]
        blit_frame(screen, display_frame)

        if state in ("edit_typing", "edit_generating"):
            draw_dialogue(screen, font, typed_text, state == "edit_generating")
        elif state in ("edit_idle", "edit_positioning"):
            draw_edit_hud(screen, font, state == "edit_positioning")

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
