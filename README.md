# SOMA Retargeter
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

![SOMA Retargeter Banner](assets/docs/banner.gif)

Convert [SOMA](https://github.com/NVlabs/SOMA-X) human motion captures into humanoid robot joint animation. Takes BVH motion files as input and produces robot-playable CSV joint data as output using GPU-optimized inverse kinematics via [Newton](https://github.com/newton-physics/newton) and high-performance computation with [NVIDIA Warp](https://github.com/NVIDIA/warp).

The retargeting pipeline handles proportional human-to-robot scaling, multi-objective IK solving with joint limits, feet stabilization to maintain ground contact, and per-DOF joint limit clamping. Currently supports SOMA as the input skeleton and Unitree G1 (29 DOF) as the output robot. Additional robot targets are planned.

SOMA Retargeter is part of the [SOMA body model](https://github.com/NVlabs/SOMA-X) ecosystem for humanoid motion data.

> **Note:** This project is in active development. The API may change between releases as the design is refined.

## Requirements

- **Python:** 3.12
- **Git LFS:** Installed and initialized for asset downloads
- **OS:** Windows (x86-64) and Linux (x86-64, aarch64)
- **GPU:** NVIDIA GPU (Maxwell or newer), driver 545+ (CUDA 12). No local CUDA Toolkit installation required.

## Installation

<details>

<summary>Setup instructions</summary>

### Method 1 (conda + pip)

#### 1. Create and Activate Conda Environment

```bash
conda create -n soma-retargeter python=3.12 -y
conda activate soma-retargeter
```

#### 2. Download LFS Assets

```bash
git lfs pull
```

#### 3. Install the Library

```bash
pip install .
```

### Method 2 (uv)

#### 1. Install uv

Follow the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/) if `uv` is not yet installed.

#### 2. Download LFS Assets

```bash
git lfs pull
```

#### 3. Sync the Project

`uv sync` creates an isolated `.venv` virtual environment inside the project directory, installs the correct Python version and resolves all dependencies.

```bash
uv sync
```

### Platform-specific notes

**Note (Linux):** For the GUI viewer to work, install `tkinter`

```bash
sudo apt-get install python3.12-tk
```

**Note (Windows):** If `imgui-bundle` fails to install, the Microsoft Visual C++ Redistributables may be missing. Download from the [official Microsoft documentation](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

</details>

## Motion Data

This repo includes 10 sample BVH/CSV pairs in `assets/motions/` for immediate testing.

For large-scale motion data, see the [SEED dataset](https://huggingface.co/datasets/bones-studio/seed) (Skeletal Everyday Embodiment Dataset) published by [Bones Studio](https://huggingface.co/bones-studio). SEED provides a large-scale collection of human motions on the SOMA uniform-proportion skeleton, which is the expected input format for this tool. The G1 robot motion data included in SEED was retargeted using SOMA Retargeter.

## Quick Start

> When using **uv** (Method 2), replace `python` with `uv run` in the commands below.

### Interactive viewer (OpenGL)

```bash
python ./app/bvh_to_csv_converter.py --config ./assets/default_bvh_to_csv_converter_config.json --viewer gl
```

![Interactive viewer interface](assets/docs/interactive-viewer-screenshot.png)

The default config now opens the viewer with the local Tara robot in the 3D viewport. BVH playback, retargeting, CSV playback, and CSV export work there as well.

To switch the interactive viewer back to the original G1 retarget workflow, set `"viewer_robot": "unitree_g1"` and `"retarget_target": "unitree_g1"` in [assets/default_bvh_to_csv_converter_config.json](/home/jony/soma-retargeter/assets/default_bvh_to_csv_converter_config.json).

### Robot asset viewer

Load a robot model directly into the simulation environment without running the retargeting pipeline.

```bash
python ./app/robot_model_viewer.py --robot tara --viewer gl
```

The viewer keeps the final Tara build under `tara` and can also load the existing G1 asset:

```bash
python ./app/robot_model_viewer.py --robot unitree_g1 --viewer gl
```

To open any other local MJCF file, pass it explicitly:

```bash
python ./app/robot_model_viewer.py --mjcf /absolute/path/to/robot.xml --viewer gl
```

### Batch conversion (headless)

Process a folder of BVH files without a display. Set `import_folder` and `export_folder` in the config file, then run:

```bash
python ./app/bvh_to_csv_converter.py --config ./assets/default_bvh_to_csv_converter_config.json --viewer null
```

Batch mode recursively finds all `.bvh` files in the import folder, processes them in configurable batch sizes, and writes CSV files to the export folder mirroring the input directory structure.

## Code Overview

### `app/`

| File | Description |
|------|-------------|
| `bvh_to_csv_converter.py` | Main entry point. Drives both interactive and headless batch retargeting modes. |
| `robot_model_viewer.py` | Minimal simulation viewer for loading Tara, G1, or any local MJCF robot asset. |

### `soma_retargeter/`

| Module | Description |
|--------|-------------|
| `animation/` | Core data structures for skeletons, animation buffers, IK, and skinned meshes. |
| `assets/` | File I/O for BVH, CSV, and USD formats. |
| `pipelines/` | Retargeting pipeline: IK solving, feet stabilization, and joint limit clamping. |
| `robotics/` | Human-to-robot scaling and robot output formatting. |
| `renderers/` | Visualization for the interactive viewer. |
| `utils/` | Math, pose, coordinate conversion, Newton and Warp helpers. |
| `configs/` | JSON configuration for retargeting, scaling, and feet stabilization parameters. |

## Related Work

SOMA Retargeter is a support tool within the SOMA ecosystem for humanoid motion data:

* [SOMA Body Model](https://github.com/NVlabs/SOMA-X) - Parametric human body model with standardized skeleton, mesh, and shape parameters
* [GEM-X](https://github.com/NVlabs/GEM-X) - Human motion estimation from video
* [Kimodo](https://github.com/nv-tlabs/kimodo) - Kinematic motion diffusion model for text and constraint-driven 3D human and robot motion generation
* [ProtoMotions](https://github.com/NVlabs/ProtoMotions) - GPU-accelerated simulation and learning framework for training physically simulated digital humans and humanoid robots
* [SONIC](https://nvlabs.github.io/GEAR-SONIC/) - Whole-body control for humanoid robots, training locomotion and interaction policies

## Acknowledgments

This project draws inspiration and builds upon excellent open-source work, including:
* [GMR](https://github.com/YanjieZe/GMR) - General Motion Retargeting
* [PyRoki](https://pyroki-toolkit.github.io/) - A Modular Toolkit for Robot Kinematic Optimization

## License

This codebase is licensed under [Apache-2.0](LICENSE).

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.
