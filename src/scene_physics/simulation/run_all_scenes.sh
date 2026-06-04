#!/usr/bin/env bash
#
# Run sim_sampling.py over every generated scene in
# resources/generated_scenes/.
#
# Usage:
#   ./run_all_scenes.sh                  # run all scenes
#   ./run_all_scenes.sh scene005 scene042   # run only the named scenes
#
# Honors the same env vars as sim_sampling.py (NUM_WORLDS, NUM_EPOCHS).
# Continues past a scene that errors and prints a summary at the end.

set -u

# --- locate paths (anchored to this script, not the cwd) ---------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../simulation
PKG_DIR="$(dirname "$SCRIPT_DIR")"                           # .../src/scene_physics
PROJECT_ROOT="$(cd "$PKG_DIR/../.." && pwd)"                 # .../Scene-Physics
SCENES_DIR="$PROJECT_ROOT/resources/generated_scenes"

if [[ ! -d "$SCENES_DIR" ]]; then
    echo "error: scenes dir not found: $SCENES_DIR" >&2
    exit 1
fi

# --- pick scenes: explicit args, else every scene dir with a physics USD ------
if [[ $# -gt 0 ]]; then
    scenes=("$@")
else
    scenes=()
    for d in "$SCENES_DIR"/*/; do
        name="$(basename "$d")"
        if [[ -f "$d/data/${name}_physics.usdc" ]]; then
            scenes+=("$name")
        fi
    done
fi

total=${#scenes[@]}
if [[ $total -eq 0 ]]; then
    echo "error: no scenes to run under $SCENES_DIR" >&2
    exit 1
fi

echo "Running sim_sampling.py over $total scene(s) from $SCENES_DIR"
echo

# sim_sampling.py resolves scene paths relative to its package; run from there.
cd "$PKG_DIR"

failed=()
i=0
for name in "${scenes[@]}"; do
    i=$((i + 1))
    echo "==================================================================="
    echo "[$i/$total] $name"
    echo "==================================================================="
    if uv run simulation/sim_sampling.py "$name"; then
        echo "[$i/$total] $name: OK"
    else
        status=$?
        echo "[$i/$total] $name: FAILED (exit $status)" >&2
        failed+=("$name")
    fi
    echo
done

# --- summary -----------------------------------------------------------------
echo "==================================================================="
echo "Done: $((total - ${#failed[@]}))/$total succeeded."
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "Failed scenes: ${failed[*]}"
    exit 1
fi
