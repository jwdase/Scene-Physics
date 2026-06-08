"""Procedural material *data* for the dataset objects (pure data, no bpy).

Separated from _materials.py (which builds the node graphs and needs bpy) so the
uv-env test suite can check dataset coverage without launching Blender.

base_color is RGBA in [0,1]; bump = (kind, scale, strength) with
kind in {"wood", "noise", "voronoi"}. See _materials.build_material for how each
field maps onto a Principled BSDF. The source meshes have no UVs, so there are no
image textures here -- only solid colors + Generated-coordinate procedurals.

COLOR POLICY: every dataset object gets a UNIQUE, well-separated hue from an
18-entry categorical palette (max separability for eval/segmentation), while the
surface *finish* (roughness/metallic/transmission/coat/bump) stays physically
plausible -- so objects shade like real materials in deliberately distinct colors.
The table + floor (substrate, see _lighting.FLOOR_TEX / dining_room_table) stay
realistic wood/tile. Palette index is noted per object so collisions are auditable.
"""

from __future__ import annotations

DEFAULT = {"base_color": (0.6, 0.6, 0.6, 1.0), "roughness": 0.5}

MATERIAL_SPECS: dict[str, dict] = {
    # --- furniture / wood (realistic substrate; NOT part of the categorical palette) ---
    "dining_room_table": {
        "textures": {
            "set": "Wood063",
            "prefix": "Wood063_4K-JPG",
            "scale": 1.2,
            "bump": 0.45,
            "coat": 0.0,          # oiled matte wood -- no clearcoat mirror layer
            "rough_floor": 0.42,  # lift the roughness-map lows so the top can't blow
                                  # out into a wet-plastic specular hotspot
        }
    },
    # --- wood blocks (finish kept matte; categorical colors) ---
    "square_wood_block": {
        "base_color": (0.90, 0.10, 0.10, 1),  # palette 01 red
        "roughness": 0.45,
        "bump": ("wood", 16.0, 0.12),
    },
    "star_wood_block": {
        "base_color": (0.40, 0.55, 0.08, 1),  # palette 17 olive
        "roughness": 0.45,
        "bump": ("wood", 16.0, 0.12),
    },
    "bung": {
        "base_color": (0.45, 0.26, 0.10, 1),  # palette 14 brown
        "roughness": 0.8,
        "bump": ("noise", 40.0, 0.20),
    },  # cork
    # --- ceramics / glossy dielectrics (satin glaze finish; categorical colors) ---
    "jug04": {"base_color": (0.10, 0.70, 0.95, 1), "roughness": 0.30, "coat": 0.1},   # palette 07 cyan
    "vase_05": {"base_color": (0.62, 0.12, 0.85, 1), "roughness": 0.28, "coat": 0.1},  # palette 10 purple
    "coffeemug": {"base_color": (0.10, 0.70, 0.20, 1), "roughness": 0.30, "coat": 0.1},  # palette 05 green
    "int_kitchen_accessories_le_creuset_bowl_30cm": {
        "base_color": (0.88, 0.12, 0.78, 1),  # palette 11 magenta
        "roughness": 0.28,
        "coat": 0.12,
    },  # enameled, satin glaze
    # --- glass / metal (transmission & metallic preserved; categorical tint) ---
    "glass1": {
        "base_color": (0.12, 0.30, 0.90, 1),  # palette 08 blue tint
        "roughness": 0.16,                    # frosted -> the form reads, depth is legible
        "transmission": 0.82,                 # murky, not perfectly clear
        "ior": 1.5,
    },
    "b04_candle_holder_metal": {
        "base_color": (0.80, 0.62, 0.08, 1),  # palette 15 gold (tinted metal)
        "metallic": 1.0,
        "roughness": 0.42,  # brushed, not chrome
    },
    "b05_coffee_grinder": {
        "base_color": (0.35, 0.15, 0.82, 1),  # palette 09 indigo (tinted metal)
        "metallic": 0.6,
        "roughness": 0.48,
    },
    "f10_apple_iphone_4": {
        "base_color": (0.30, 0.45, 0.65, 1),  # palette 16 slate-blue
        "roughness": 0.22,
        "coat": 0.4,
    },  # glossy slab
    "round_coaster_stone": {
        "base_color": (0.10, 0.42, 0.30, 1),  # palette 18 deep-green
        "roughness": 0.7,
        "bump": ("noise", 50.0, 0.25),
    },
    # --- organic (matte/SSS finish; categorical colors) ---
    "b03_loafbread": {
        "base_color": (0.95, 0.50, 0.05, 1),  # palette 02 orange
        "roughness": 0.85,
        "subsurface": 0.15,
        "bump": ("noise", 28.0, 0.30),
    },
    "banana_fix2": {
        "base_color": (0.95, 0.85, 0.10, 1),  # palette 03 yellow
        "roughness": 0.45,
        "subsurface": 0.10,
        "bump": ("noise", 18.0, 0.10),
    },
    "pepper": {
        "base_color": (0.65, 0.05, 0.20, 1),  # palette 13 crimson
        "roughness": 0.28,
        "subsurface": 0.20,
        "bump": ("noise", 12.0, 0.08),
    },
    # --- figurative (solid color; categorical colors) ---
    "shark": {"base_color": (0.05, 0.65, 0.60, 1), "roughness": 0.55},   # palette 06 teal
    "bee": {"base_color": (0.55, 0.80, 0.10, 1), "roughness": 0.45},     # palette 04 chartreuse
    "heart": {"base_color": (0.95, 0.35, 0.60, 1), "roughness": 0.40},   # palette 12 pink
}
