# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Scene-Physics investigates whether probabilistic models can reconstruct 3D scene structure while incorporating physical properties and physics simulations. The research question: Can we build probabilistic models that take physical properties and physics into account when reconstructing the 3D structure of a scene from visual input?

## Running Code

The PYTHONPATH must be set to the project root for imports to work:

```bash
PYTHONPATH=. python src/Scene-Physics/simulation/scene01.py
```

Main simulation scripts are in `src/Scene-Physics/simulation/`.

## Development Commands

```bash
# Format code
black .

# Sort imports
isort .

# Run tests
pytest
```

## Dependencies

**External Dependencies (git submodules):**
- **Newton** (`src/newton/`): NVIDIA GPU-accelerated physics engine using Warp
- **Bayes3D/b3d**: Probabilistic inverse graphics library (external, not a submodule)

**Path Setup:** Scripts must call `setup_path('jonathan')` or `setup_path('jack')` from `utils/io.py` to configure Newton and B3D paths.

## Architecture

### Core Modules (`src/Scene-Physics/`)

- **properties/**: Object representation
  - `shapes.py`: Body classes (`Sphere`, `Box`, `MeshBody`, `StableMesh`, `SoftMesh`) with position (vec3), rotation (quaternion), mass, material
  - `material.py`: Physical properties (friction, density, restitution) converting to Newton's `ShapeConfig`

- **simulation/**: Physics simulation scripts using Newton's `ModelBuilder` and `SolverXPBD`

- **visualization/**: PyVista-based rendering
  - `scene.py`: `Visualizer`, `VideoVisualizer`, `PyVistaVisualizer` classes for static/animated scene rendering and point cloud generation

- **physics/kernels.py**: Custom Warp kernels for force application

- **tasks/**: Experimental code including MCMC sampling with GenJax and B3D

- **objects/**: 3D mesh files (.obj) for simulation scenes

- **recordings/**: Output directory for rendered images, videos, and point clouds

### Key Patterns

- **Coordinate System**: Y-axis up, gravity -9.81 on Y
- **Quaternion Format**: XYZW ordering (Warp/Scipy convention)
- **Position/Rotation Updates**: Must explicitly update `builder.body_q` and `builder.shape_transform`
- **JAX Integration**: Use pure callbacks to bridge imperative Newton physics code with functional JAX/GenJax
- **Material Defaults**: Dynamic objects use density 1000.0 kg/m³, friction 0.5, restitution 0.0; static objects use density 0.0

### Probabilistic Inference

GenJax models define probabilistic processes for scene reconstruction. MCMC sampling (Metropolis-Hastings) over object positions uses B3D likelihood computation for visual observations via point cloud matching.
