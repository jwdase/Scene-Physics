#!/usr/bin/env bash
# Fetch the large render assets that are gitignored (CC0 textures + HDRI), so the
# Blender render pipeline can find them. Run from anywhere:
#     bash resources/fetch_assets.sh
#
# Textures: ambientCG (CC0). We keep only the maps the materials read
# (Color / Roughness / Displacement). HDRI: Poly Haven (CC0).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEX="$ROOT/resources/textures"
HDRI="$ROOT/resources/hdri"
UA="Mozilla/5.0 (X11; Linux x86_64)"
mkdir -p "$TEX" "$HDRI"

fetch_acg() { # $1 = ambientCG asset id (e.g. Wood063)
  local id="$1"
  echo "ambientCG: $id (4K-JPG)"
  curl -fsSL -A "$UA" -o "/tmp/${id}.zip" "https://ambientcg.com/get?file=${id}_4K-JPG.zip"
  rm -rf "$TEX/$id"
  unzip -oq "/tmp/${id}.zip" -d "$TEX/$id"
  # Drop maps the materials don't use + descriptor files.
  find "$TEX/$id" -type f \
    ! \( -iname "*_Color.jpg" -o -iname "*_Roughness.jpg" -o -iname "*_Displacement.jpg" \) -delete
  rm -f "/tmp/${id}.zip"
}

fetch_hdri() { # $1 = Poly Haven slug (e.g. lythwood_room)
  local name="$1"
  echo "PolyHaven: ${name}_4k.hdr"
  curl -fsSL -A "$UA" -o "$HDRI/${name}_4k.hdr" \
    "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/4k/${name}_4k.hdr"
}

fetch_acg Wood063     # table surface
fetch_acg Tiles108    # floor tile grid
fetch_hdri lythwood_room  # interior environment / key light

echo "Done. Assets in resources/textures/ and resources/hdri/."
