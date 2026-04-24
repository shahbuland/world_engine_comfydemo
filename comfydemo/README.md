# comfydemo

A live, playable demo built on top of [Overworld's world_engine](https://github.com/Overworldai/world_engine). The model (`Waypoint-1.5-1B-360P`) generates game frames in real time conditioned on your keyboard and mouse input. You can pause at any point, send the current frame through any image transformation pipeline, and continue playing from the modified frame.

---

## Quick start

```sh
# From the repo root
pip install -r comfydemo/requirements.txt
pip install --upgrade "world_engine @ git+https://github.com/Overworldai/world_engine.git"

export HF_TOKEN=<your Hugging Face token>  # needs access to Overworld/Waypoint-1.5-1B-360P

python comfydemo/play.py
```

### Controls

| Key / Input | Action |
|---|---|
| `W A S D` | Move |
| Mouse | Look |
| `Space` | Jump / action |
| `Left click` | Primary action |
| `Scroll wheel` | Scroll input |
| `P` | Pause / resume |
| `T` | Trigger frame hook (while paused) |
| `Escape` | Quit |

---

## How it works

### The world model

`Waypoint-1.5` is a video generation model that acts like a game engine: given a history of frames and a controller input, it predicts the next frame. It generates frames in batches of 4 (temporal compression), so each call to `gen_frame()` returns `[4, 360, 640, 3]` uint8 numpy array at 360p.

The model keeps a KV cache of everything it has seen, which is what makes it feel stateful — it "remembers" what happened earlier in the session.

### Files

**`engine_loop.py` — `EngineLoop`**

Wraps `WorldEngine` and runs `gen_frame()` in a background thread so the pygame window never blocks. Key methods:

- `start(seed_frames)` — kick off generation from a `[4, H, W, 3]` uint8 seed
- `set_ctrl(CtrlInput)` — update the controller state (called every frame from the render loop)
- `pause()` / `resume()` — freeze/unfreeze the generation thread
- `inject_frames(frames)` — push a `[4, H, W, 3]` uint8 array into the model's KV cache; subsequent generation continues from this frame. **Must be called while paused.**
- `frame_queue` — `queue.Queue` of `[4, H, W, 3]` uint8 numpy arrays; the render loop drains this

**`frame_hooks.py` — `FrameHook`**

A hook is anything that takes `[4, H, W, 3]` uint8 frames and returns `[4, H, W, 3]` uint8 frames. The base class is a passthrough. Subclass it to add any transformation.

- `FrameHook` — base class, override `process(frames) -> frames`
- `HTTPImg2ImgHook(url)` — POSTs the last frame as a PNG to any HTTP endpoint, expects a PNG back, tiles the result across all 4 frames

Hooks run in a background thread (started by `play.py`) so the window stays responsive while waiting for a slow API.

**`play.py` — pygame app**

The render loop. Handles input, drains `frame_queue`, blits frames, and wires up the hook to `T`. Pass `--hook <url>` to activate an `HTTPImg2ImgHook`.

---

## Adding your own hook

Subclass `FrameHook` in `frame_hooks.py`:

```python
class MyHook(FrameHook):
    def process(self, frames: np.ndarray) -> np.ndarray:
        # frames: [4, H, W, 3] uint8
        # do anything here — call an API, run a local model, apply a PIL filter, etc.
        # return [4, H, W, 3] uint8
        last_frame = frames[-1]  # most recent frame — usually the interesting one
        modified = do_something(last_frame)
        return np.stack([modified] * 4).astype(np.uint8)
```

Then instantiate it in `play.py` and pass it to the game loop instead of the default `FrameHook()`.

---

## The image2image demo flow

The intended demo:

1. **Play normally** — WASD around, the world generates in real time
2. **Press `P`** — generation freezes on the current frame
3. **Press `T`** — the last frame is sent to your hook; a status overlay appears while it runs
4. **Hook returns a modified frame** — injected into the model's KV cache via `inject_frames()`
5. **Press `P` to resume** — generation continues from the modified frame as if it always looked that way

Because `inject_frames` appends to the KV cache rather than replacing it, the model sees the modified frame as the latest frame in its history. The further you play after an injection, the more the model "commits" to the new look.

---

## For Claude: context for helping users extend this

If you're a Claude instance helping someone build on this demo, here's the full picture:

**What `WorldEngine` exposes (from `src/world_engine.py`):**
- `WorldEngine(model_uri, device, quant)` — loads model, VAE, KV cache
- `engine.set_prompt(str)` — text conditioning (raises if the model doesn't support it; safe to try/except)
- `engine.append_frame(tensor)` — encodes a `[4, H, W, 3]` uint8 torch tensor into the KV cache without generating
- `engine.gen_frame(ctrl=CtrlInput)` — denoises a new frame, caches it, returns `[4, H, W, 3]` uint8 tensor
- `engine.get_state()` / `engine.load_state(state)` — snapshot and restore the full KV cache state
- `engine.reset()` — wipe KV cache and start fresh
- `engine.device` — the `torch.device` the model lives on

**`CtrlInput` fields:**
- `button: Set[int]` — set of pressed Owl-Control keycodes (uppercase ASCII for letter keys: W=87, A=65, S=83, D=68, Space=32)
- `mouse: Tuple[float, float]` — `(dx, dy)` relative mouse movement, normalized (small values like 0.01–0.3)
- `scroll_wheel: int` — `-1`, `0`, or `1`

**Things worth knowing:**
- The 360P model outputs `[4, 360, 640, 3]`. The 720P models output `[4, 720, 1280, 3]`.
- `append_frame` asserts `dtype == torch.uint8` — if you're passing modified frames, make sure they're uint8.
- The KV cache grows unbounded during a session. Very long sessions may slow down or run out of VRAM. `engine.reset()` clears it.
- `get_state()` / `load_state()` are useful for saving a "checkpoint" before an injection so you can revert if the modification goes wrong.
- The generation thread in `EngineLoop` reads `self._ctrl` under a lock each iteration — so controller updates from the render loop are naturally rate-limited to generation speed.

**Ideas for cool extensions:**
- Save/load world states with `get_state()` / `load_state()` — lets you branch timelines
- Apply a local PIL/torchvision filter in a hook without any API call (sepia, blur, edge detection, etc.)
- Run a local diffusion model (e.g. via diffusers) as the hook for true style transfer
- Log `last_frames` to disk on pause to build a highlight reel
- Swap the prompt mid-session with `engine.set_prompt()` and observe the world shift
