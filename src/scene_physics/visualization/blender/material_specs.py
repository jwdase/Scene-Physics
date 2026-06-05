"""Procedural material *data* for the dataset objects (pure data, no bpy).

Separated from _materials.py (which builds the node graphs and needs bpy) so the
uv-env test suite can check dataset coverage without launching Blender.

base_color is RGBA in [0,1]; bump = (kind, scale, strength) with
kind in {"wood", "noise", "voronoi"}. See _materials.build_material for how each
field maps onto a Principled BSDF. The source meshes have no UVs, so there are no
image textures here -- only solid colors + Generated-coordinate procedurals.
"""

from __future__ import annotations

DEFAULT = {"base_color": (0.6, 0.6, 0.6, 1.0), "roughness": 0.5}

MATERIAL_SPECS: dict[str, dict] = {
    # --- furniture / wood ---
    "dining_room_table": {
        "base_color": (0.17, 0.085, 0.04, 1),
        "roughness": 0.35,
        "spec": 0.5,
        "bump": ("wood", 5.0, 0.20),
    },
    "square_wood_block": {
        "base_color": (0.55, 0.38, 0.20, 1),
        "roughness": 0.45,
        "bump": ("wood", 16.0, 0.12),
    },
    "star_wood_block": {
        "base_color": (0.58, 0.40, 0.22, 1),
        "roughness": 0.45,
        "bump": ("wood", 16.0, 0.12),
    },
    "bung": {
        "base_color": (0.66, 0.50, 0.32, 1),
        "roughness": 0.8,
        "bump": ("noise", 40.0, 0.20),
    },  # cork
    # --- ceramics / glossy dielectrics ---
    "jug04": {"base_color": (0.85, 0.82, 0.75, 1), "roughness": 0.15},
    "vase_05": {"base_color": (0.80, 0.80, 0.83, 1), "roughness": 0.12},
    "coffeemug": {"base_color": (0.90, 0.90, 0.92, 1), "roughness": 0.12},
    "int_kitchen_accessories_le_creuset_bowl_30cm": {
        "base_color": (0.74, 0.12, 0.05, 1),
        "roughness": 0.08,
    },  # enameled red
    # --- glass / metal ---
    "glass1": {
        "base_color": (1, 1, 1, 1),
        "roughness": 0.0,
        "transmission": 1.0,
        "ior": 1.45,
    },
    "b04_candle_holder_metal": {
        "base_color": (0.72, 0.72, 0.74, 1),
        "metallic": 1.0,
        "roughness": 0.30,
    },
    "b05_coffee_grinder": {
        "base_color": (0.12, 0.12, 0.13, 1),
        "metallic": 0.6,
        "roughness": 0.40,
    },
    "f10_apple_iphone_4": {
        "base_color": (0.02, 0.02, 0.03, 1),
        "roughness": 0.08,
        "coat": 1.0,
    },  # black glass slab
    "round_coaster_stone": {
        "base_color": (0.45, 0.45, 0.47, 1),
        "roughness": 0.7,
        "bump": ("noise", 50.0, 0.25),
    },
    # --- organic ---
    "b03_loafbread": {
        "base_color": (0.62, 0.42, 0.22, 1),
        "roughness": 0.85,
        "subsurface": 0.15,
        "bump": ("noise", 28.0, 0.30),
    },
    "banana_fix2": {
        "base_color": (0.85, 0.72, 0.12, 1),
        "roughness": 0.45,
        "subsurface": 0.10,
        "bump": ("noise", 18.0, 0.10),
    },
    "pepper": {
        "base_color": (0.75, 0.10, 0.08, 1),
        "roughness": 0.18,
        "subsurface": 0.20,
        "bump": ("noise", 12.0, 0.08),
    },  # red bell pepper
    # --- figurative (solid color; no decals possible without UVs) ---
    "shark": {"base_color": (0.45, 0.52, 0.58, 1), "roughness": 0.5},
    "bee": {"base_color": (0.85, 0.65, 0.05, 1), "roughness": 0.4},
    "heart": {"base_color": (0.75, 0.05, 0.10, 1), "roughness": 0.2},
}
