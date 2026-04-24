"""
Background generation loop for WorldEngine.
Runs gen_frame on the main thread (torch.compile + CUDA graphs are not thread-safe).
Auto-resets after AUTO_RESET_SECS of active (unpaused) play to avoid KV cache overflow.
"""
import time

import numpy as np
import torch

from world_engine import WorldEngine, CtrlInput

AUTO_RESET_SECS = 20.0


class EngineLoop:
    def __init__(self, model_id: str, device: str = "cuda", quant=None):
        print(f"Loading {model_id}...")
        self.engine = WorldEngine(model_id, device=device, quant=quant)
        self._device = self.engine.device

        self.last_frames: np.ndarray | None = None
        self._seed_frames: np.ndarray | None = None

        self._paused = False
        self._reset_requested = False
        self._active_secs = 0.0
        self._last_step_t: float | None = None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def set_prompt(self, prompt: str):
        try:
            self.engine.set_prompt(prompt)
        except RuntimeError:
            pass  # model has no prompt conditioning

    def start(self, seed_frames: np.ndarray):
        """seed_frames: [4, H, W, 3] uint8 numpy array."""
        self._seed_frames = seed_frames
        tensor = torch.from_numpy(seed_frames).to(self._device)
        self.engine.append_frame(tensor)
        self.last_frames = seed_frames

    def step(self, ctrl: CtrlInput) -> np.ndarray:
        """
        Generate one frame batch synchronously. Call from the main loop each tick.
        Returns the latest [4, H, W, 3] uint8 numpy array.
        """
        if self._paused:
            return self.last_frames

        now = time.monotonic()
        if self._last_step_t is not None:
            self._active_secs += now - self._last_step_t
        self._last_step_t = now

        if self._reset_requested or self._active_secs >= AUTO_RESET_SECS:
            self._do_reset()
            self._reset_requested = False
            self._active_secs = 0.0
            return self.last_frames

        frames = self.engine.gen_frame(ctrl=ctrl).cpu().numpy()
        self.last_frames = frames
        return frames

    def reset_scene(self):
        self._reset_requested = True

    def reseed(self, seed_frames: np.ndarray):
        """Swap to a new seed image and reset immediately."""
        self._seed_frames = seed_frames
        self._reset_requested = True

    def restart_from(self, frames: np.ndarray):
        """
        Drop all KV cache context and restart generation from the given frames.
        frames: [4, H, W, 3] uint8 numpy array.
        Stores the VAE-decoded version as last_frames so the caller can see
        the compression artefacts confirming the frame went through the model.
        Call from the main thread (e.g. after compositing HUD elements).
        """
        self.engine.reset()
        self._seed_frames = frames
        tensor = torch.from_numpy(frames).to(self._device)
        decoded = self.engine.append_frame(tensor)   # [4, H, W, 3] uint8 tensor
        self.last_frames = decoded.cpu().numpy()
        self._active_secs = 0.0
        self._last_step_t = None
        self._reset_requested = False

    def pause(self):
        self._paused = True
        self._last_step_t = None  # don't count paused time toward auto-reset

    def resume(self):
        self._paused = False

    @property
    def is_paused(self) -> bool:
        return self._paused

    def inject_frames(self, frames: np.ndarray):
        """
        Push a modified [4, H, W, 3] uint8 frame batch into the engine's KV cache
        so that subsequent generation continues from it. Call only while paused.
        """
        tensor = torch.from_numpy(frames).to(self._device)
        self.engine.append_frame(tensor)
        self.last_frames = frames

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _do_reset(self):
        print("Resetting scene...")
        self.engine.reset()
        tensor = torch.from_numpy(self._seed_frames).to(self._device)
        self.engine.append_frame(tensor)
        self.last_frames = self._seed_frames
