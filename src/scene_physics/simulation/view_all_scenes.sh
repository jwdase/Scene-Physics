#!/usr/bin/env bash
# Run simulation.py once per generated scene, spawning a fresh process each time.
# The window must be closed before the next scene loads (avoids Newton/GPU state leaks).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"  # src/scene_physics/

cd "$REPO_ROOT"

SCENES_DIR="generated_scenes"
TOTAL=$(seq -f "%03g" 1 100 | while read -r n; do
    if [ -d "$SCENES_DIR/scene$n" ]; then echo "$SCENES_DIR/scene$n"; fi
done | wc -l)
COUNT=0

for n in $(seq -f "%03g" 1 100); do
    scene_dir="$SCENES_DIR/scene$n"
    # (original glob guard replaced by explicit range)
    [ -d "$scene_dir" ] || continue

    scene_name="scene$n"
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
