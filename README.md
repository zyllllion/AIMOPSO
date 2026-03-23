# AIMOPSO (Visualization-First Reviewer Release)

This repository provides a reviewer-facing, minimal public release for:

**A*-Guided Heuristic Multi-Objective PSO for UAV 3D Path Planning**.

The recommended entrypoint is:

```bash
python visualize_aimopso_diverse_solutions.py
```

## Quick Start

```bash
pip install -r requirements.txt
python visualize_aimopso_diverse_solutions.py
```

### Windows PowerShell Example

```powershell
$env:SCENE_TO_RUN='1'
$env:SEED='42'
python visualize_aimopso_diverse_solutions.py
```

## Dependency Note (Mayavi)

In some Windows environments, `pip install mayavi` may fail due compiler/VTK wheel constraints.
If this happens, install Mayavi via conda and keep the rest via pip:

```bash
conda install -c conda-forge mayavi vtk
pip install -r requirements.txt
```

## Output

By default, the visualization script writes figures to:

- `aimopso_diverse_solutions_fair30/`

## Reviewer Figure

- `assets/reviewer/fig01_02_aimopso_s1_s4_stacked.png`

## Public Contents in This Release

- Visualization entry script: `visualize_aimopso_diverse_solutions.py`
- Core runtime chain required by visualization:
  - `aimopso_runner.py`
  - `aimopso_operators.py`
  - `a_star_guidance.py`
  - `coordinates.py`
  - `cost_function.py`
  - `environments.py`
  - `pso_operators.py`
  - `plotting_matlab_exact_final2.py`
  - `algorithm_cache_manager.py`
  - `feasibility.py`

## Note on Full Release

Complete large-scale benchmark pipelines, full analysis scripts, and full experiment datasets will be released after paper acceptance.
