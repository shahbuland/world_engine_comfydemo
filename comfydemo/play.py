"""
Live playable Waypoint-1.5 demo.

Controls:
  A-Z           all letter keys
  1-4           number keys
  Arrow keys    directional input
  Tab / Space   action keys
  L/R Ctrl      modifier
  Mouse         look
  Left click    primary action
  Scroll        scroll wheel input
  P             pause / resume
  7             reset scene to seed frame
  T             trigger frame hook (while paused)
  Escape        quit

Usage:
  python play.py
  python play.py --model Overworld/Waypoint-1.5-1B-360P --prompt "A sunny forest"
  python play.py --hook http://localhost:5000/img2img
"""
import argparse
import sys
import os
import threading
import random

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

import numpy as np
import cv2
import pygame

# Allow running from repo root or from comfydemo/
sys.path.insert(0, os.path.dirname(__file__))
from engine_loop import EngineLoop
from frame_hooks import FrameHook, HTTPImg2ImgHook

# ---------------------------------------------------------------------------
# Owl-Control keycode mapping
# Letters -> uppercase ASCII; special keys -> Windows Virtual-Key codes
# See: https://github.com/Overworldai/owl-control/blob/main/src/system/keycode.rs
# ---------------------------------------------------------------------------
KEY_MAP = {
    # Letters (pygame lowercase -> Owl-Control uppercase ASCII)
    **{getattr(pygame, f"K_{c}"): ord(c.upper()) for c in "abcdefghijklmnopqrstuvwxyz"},
    # Numbers 1-4
    pygame.K_1: 49, pygame.K_2: 50, pygame.K_3: 51, pygame.K_4: 52,
    # Common keys
    pygame.K_SPACE:  32,
    pygame.K_TAB:    9,
    pygame.K_LCTRL:  162,
    pygame.K_RCTRL:  163,
    # Arrow keys (Windows VK codes)
    pygame.K_LEFT:   37,
    pygame.K_UP:     38,
    pygame.K_RIGHT:  39,
    pygame.K_DOWN:   40,
}

MOUSE_LEFT_BTN = 1   # Owl-Control VK codes for mouse buttons
MOUSE_RIGHT_BTN = 2
MOUSE_MIDDLE_BTN = 4
RESOLUTION_360P = (640, 360)
DISPLAY_RESOLUTION = (1280, 720)
MOUSE_SENSITIVITY = 1.0


SEEDS_DIR = os.path.join(os.path.dirname(__file__), "seeds")


def load_seed_frame(resolution: tuple[int, int], exclude: str = None) -> np.ndarray:
    """Pick a random jpg from comfydemo/seeds/ and resize to target resolution."""
    paths = [p for p in os.listdir(SEEDS_DIR) if p.lower().endswith(".jpg") and p != exclude]
    chosen = os.path.join(SEEDS_DIR, random.choice(paths))
    frame = cv2.imdecode(np.fromfile(chosen, np.uint8), cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.resize(frame, resolution)


def blit_frame(screen: pygame.Surface, frame: np.ndarray):
    """Draw a [H, W, 3] uint8 numpy frame to the pygame surface, stretched to display size."""
    surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
    screen.blit(pygame.transform.scale(surf, DISPLAY_RESOLUTION), (0, 0))


def draw_overlay(screen: pygame.Surface, font: pygame.font.Font, state: str):
    if state == "paused":
        text = "PAUSED  |  [T] send to hook  |  [P] resume"
        label = font.render(text, True, (255, 220, 0))
        screen.blit(label, (10, 10))
    elif state == "hook_running":
        label = font.render("Sending to hook...", True, (100, 200, 255))
        screen.blit(label, (10, 10))


def main():
    parser = argparse.ArgumentParser(description="Live Waypoint-1.5 demo")
    parser.add_argument("--model", default="Overworld/Waypoint-1.5-1B-360P")
    parser.add_argument("--prompt", default=None, help="Text prompt (optional)")
    parser.add_argument("--hook", default=None, metavar="URL",
                        help="HTTP img2img endpoint to call on [T] press")
    args = parser.parse_args()

    hook: FrameHook = HTTPImg2ImgHook(args.hook) if args.hook else FrameHook()

    # --- Engine ---
    loop = EngineLoop(args.model)
    if args.prompt:
        loop.set_prompt(args.prompt)

    seed = load_seed_frame(RESOLUTION_360P)
    seed_x4 = np.stack([seed] * 4)  # [4, H, W, 3]

    # --- Pygame ---
    pygame.init()
    screen = pygame.display.set_mode(DISPLAY_RESOLUTION)
    pygame.display.set_caption("Waypoint-1.5")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)

    blit_frame(screen, seed)
    pygame.display.flip()

    loop.start(seed_x4)

    pressed: set[int] = set()
    scroll = 0
    current_frames = seed_x4
    hook_busy = False
    reseed_busy = False

    def run_hook():
        nonlocal hook_busy, current_frames
        hook_busy = True
        try:
            modified = hook.process(current_frames)
            loop.inject_frames(modified)
            current_frames = modified
        except Exception as e:
            print(f"Hook error: {e}")
        hook_busy = False

    def run_reseed():
        nonlocal reseed_busy
        reseed_busy = True
        try:
            new_seed = load_seed_frame(RESOLUTION_360P)
            loop.reseed(np.stack([new_seed] * 4))
        except Exception as e:
            print(f"Reseed error: {e}")
        reseed_busy = False

    running = True
    while running:
        scroll = 0
        mouse_dx = mouse_dy = 0.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p:
                    if loop.is_paused:
                        loop.resume()
                    else:
                        loop.pause()
                elif event.key == pygame.K_7:
                    loop.reset_scene()
                elif event.key == pygame.K_8 and not reseed_busy:
                    threading.Thread(target=run_reseed, daemon=True).start()
                elif event.key == pygame.K_t and loop.is_paused and not hook_busy:
                    threading.Thread(target=run_hook, daemon=True).start()
                owl = KEY_MAP.get(event.key)
                if owl is not None:
                    pressed.add(owl)

            elif event.type == pygame.KEYUP:
                owl = KEY_MAP.get(event.key)
                if owl is not None:
                    pressed.discard(owl)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    pressed.add(MOUSE_LEFT_BTN)
                elif event.button == 3:
                    pressed.add(MOUSE_RIGHT_BTN)
                elif event.button == 2:
                    pressed.add(MOUSE_MIDDLE_BTN)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    pressed.discard(MOUSE_LEFT_BTN)
                elif event.button == 3:
                    pressed.discard(MOUSE_RIGHT_BTN)
                elif event.button == 2:
                    pressed.discard(MOUSE_MIDDLE_BTN)

            elif event.type == pygame.MOUSEMOTION:
                mouse_dx += event.rel[0] * MOUSE_SENSITIVITY
                mouse_dy += event.rel[1] * MOUSE_SENSITIVITY

            elif event.type == pygame.MOUSEWHEEL:
                scroll = 1 if event.y > 0 else -1

        from world_engine import CtrlInput
        ctrl = CtrlInput(button=set(pressed), mouse=[mouse_dx, mouse_dy], scroll_wheel=scroll)
        current_frames = loop.step(ctrl)

        blit_frame(screen, current_frames[-1])

        if hook_busy:
            draw_overlay(screen, font, "hook_running")
        elif loop.is_paused:
            draw_overlay(screen, font, "paused")

        pygame.display.flip()
        clock.tick(60)  # only matters while paused; gen_frame naturally limits speed

    pygame.quit()


if __name__ == "__main__":
    main()
