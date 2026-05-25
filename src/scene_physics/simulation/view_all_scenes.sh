#!/usr/bin/env bash
# Run simulation.py once per generated scene, spawning a fresh process each time.
# The window must be closed before the next scene loads (avoids Newton/GPU state leaks).
#
# Reads the dataset from <repo>/resources/generated_scenes/. The path is resolved
# relative to this script, so it always targets that directory regardless of where you
# launch the script from.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"        # src/scene_physics/
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"      # Scene-Physics/
SCENES_DIR="$REPO_ROOT/resources/generated_scenes"

if [ ! -d "$SCENES_DIR" ]; then
    echo "No scenes directory at $SCENES_DIR" >&2
    echo "  (copy a generated dataset into resources/generated_scenes/ first)" >&2
    exit 1
fi

cd "$PROJECT_DIR"   # so `uv run simulation/simulation.py` and scene-relative paths resolve

# Enumerate the scenes actually present (sorted) rather than assuming a fixed count.
mapfile -t scene_dirs < <(find "$SCENES_DIR" -maxdepth 1 -type d -name 'scene*' | sort)
TOTAL=${#scene_dirs[@]}

if [ "$TOTAL" -eq 0 ]; then
    echo "No scene* directories found in $SCENES_DIR." >&2
    exit 1
fi

COUNT=0
for scene_dir in "${scene_dirs[@]}"; do
    scene_name="$(basename "$scene_dir")"
    scene_usd="$scene_dir/data/${scene_name}_physics.usdc"
    output_path="$scene_dir/data/${scene_name}_recording.usdc"

    if [ ! -f "$scene_usd" ]; then
        echo "[$scene_name] USD not found, skipping."
        continue
    fi

    COUNT=$((COUNT + 1))
    echo "[$COUNT/$TOTAL] $scene_name — close the window to continue..."

    uv run simulation/simulation.py "$scene_usd" "$output_path"
done

echo "Done — viewed $COUNT scenes."
