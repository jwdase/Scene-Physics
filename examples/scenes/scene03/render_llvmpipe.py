"""
Launch Newton on scene03 and render a video using the Mesa llvmpipe
software rasterizer.

Useful on headless nodes where GPU/EGL passthrough isn't available: CUDA
still drives Newton/Warp on the H100, while OpenGL is redirected to the
CPU via LLVM.

Run from the repo root (or anywhere — paths are resolved from __file__):

    uv run examples/scenes/scene03/render_llvmpipe.py
"""

import os

# Force Mesa software rendering through llvmpipe. Must be set BEFORE any
# OpenGL / GLFW library is imported, so it lives at the very top of the
# file before `import newton` pulls in the viewer stack.
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["GALLIUM_DRIVER"] = "llvmpipe"
os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = "llvmpipe"
os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

import numpy as np
import imageio.v2 as imageio
import warp as wp

import newton
import newton.solvers
import newton.viewer


SCENE_USD = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "objects", "scene03", "scene01.usdc",
)
OUTPUT_MP4 = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "recordings", "physics", "scene03_llvmpipe.mp4",
)

WIDTH, HEIGHT = 1280, 720
FPS = 30
DURATION_S = 4.0
SUBSTEPS = 4


def main() -> None:
    os.makedirs(os.path.dirname(OUTPUT_MP4), exist_ok=True)

    builder = newton.ModelBuilder()
    builder.add_usd(SCENE_USD)
    model = builder.finalize()

    solver = newton.solvers.SolverXPBD(model)
    state_0 = model.state()
    state_1 = model.state()
    contacts = model.collide(state_0)

    viewer = newton.viewer.ViewerGL(width=WIDTH, height=HEIGHT, headless=False)
    viewer.set_model(model)

    frame_buf = wp.zeros((HEIGHT, WIDTH, 3), dtype=wp.uint8)
    writer = imageio.get_writer(OUTPUT_MP4, fps=FPS, codec="libx264", quality=8)

    dt = 1.0 / FPS
    sub_dt = dt / SUBSTEPS
    n_frames = int(FPS * DURATION_S)

    try:
        for i in range(n_frames):
            for _ in range(SUBSTEPS):
                contacts = model.collide(state_0)
                solver.step(state_0, state_1, None, contacts, sub_dt)
                state_0, state_1 = state_1, state_0

            viewer.begin_frame(i * dt)
            viewer.log_state(state_0)
            viewer.end_frame()

            rgb = viewer.get_frame(target_image=frame_buf).numpy()
            writer.append_data(rgb)
            print(f"frame {i + 1}/{n_frames}")
    finally:
        writer.close()
        viewer.close()

    print(f"wrote {OUTPUT_MP4}")


if __name__ == "__main__":
    main()
