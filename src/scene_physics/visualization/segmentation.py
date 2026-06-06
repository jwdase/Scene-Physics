"""Decode Blender's flat-color ID render into an integer segmentation map.

Blender writes `seg_raw.png` (each object a unique emission hue, background black,
Standard/sRGB view transform) plus `seg_labels.json` (name->id and the linear
palette). Here we classify every pixel to the nearest palette color and emit:

  * segmentation.png            -- single-channel uint8, pixel value == object id
                                   (0 = background)
  * segmentation_overlay.png    -- palette colors alpha-blended over render.png
  * segmentation_labels.json    -- {name: id} (+ id 0 = background)

Nearest-color classification (rather than exact byte matching) absorbs sRGB
rounding; the hues are far enough apart that it is unambiguous.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _srgb_oetf(c: np.ndarray) -> np.ndarray:
    """Linear -> sRGB display, matching Blender's 'Standard' view transform."""
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def decode_labels(seg_raw_path: str, labels_json_path: str) -> tuple[np.ndarray, dict]:
    """Return (label_image HxW uint, {name: id})."""
    labels = json.loads(Path(labels_json_path).read_text())
    name_to_id = labels["name_to_id"]
    pal = labels["palette"]  # {id_str: [r,g,b] linear}

    ids = [0] + sorted(int(k) for k in pal)
    ref = np.zeros((len(ids), 3), dtype=np.float32)
    for j, i in enumerate(ids):
        ref[j] = (
            0.0 if i == 0 else _srgb_oetf(np.asarray(pal[str(i)], dtype=np.float32))
        )

    img = np.asarray(Image.open(seg_raw_path).convert("RGB"), dtype=np.float32) / 255.0
    h, w = img.shape[:2]
    flat = img.reshape(-1, 3)
    # nearest reference color per pixel
    d = np.linalg.norm(flat[:, None, :] - ref[None, :, :], axis=2)
    lab = np.asarray(ids, dtype=np.int32)[d.argmin(axis=1)].reshape(h, w)
    return lab.astype(np.uint8 if max(ids) < 256 else np.uint16), name_to_id


_OVERLAY_ALPHA = 0.5


def write_outputs(
    out_dir: str, seg_raw="seg_raw.png", labels="seg_labels.json", render="render.png"
) -> dict:
    """Decode + write segmentation.png, overlay, and {name:id} json. Returns name_to_id."""
    out = Path(out_dir)
    lab, name_to_id = decode_labels(str(out / seg_raw), str(out / labels))

    Image.fromarray(lab).save(out / "segmentation.png")

    # Colorized overlay over the beauty render (if present).
    pal = json.loads((out / labels).read_text())["palette"]
    color = np.zeros((*lab.shape, 3), dtype=np.float32)
    for id_str, lin in pal.items():
        disp = _srgb_oetf(np.asarray(lin, dtype=np.float32))
        color[lab == int(id_str)] = disp
    render_path = out / render
    if render_path.exists():
        base = (
            np.asarray(Image.open(render_path).convert("RGB"), dtype=np.float32) / 255.0
        )
        if base.shape[:2] == lab.shape:
            fg = lab > 0
            blended = base.copy()
            blended[fg] = (1 - _OVERLAY_ALPHA) * base[fg] + _OVERLAY_ALPHA * color[fg]
            Image.fromarray((blended * 255).astype(np.uint8)).save(
                out / "segmentation_overlay.png"
            )
    else:
        Image.fromarray((color * 255).astype(np.uint8)).save(
            out / "segmentation_overlay.png"
        )

    full = {"background": 0, **name_to_id}
    (out / "segmentation_labels.json").write_text(json.dumps(full, indent=2))
    return name_to_id
