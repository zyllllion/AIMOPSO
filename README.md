# AIMOPSO (Reviewer-Friendly Public Version)

This repository provides a **minimal reproducible implementation** for the method:

**A*-Guided Heuristic Multi-Objective PSO for UAV 3D Path Planning**.

To protect unpublished assets during peer review, only core algorithm code and lightweight demo artifacts are public in this phase.

## Public Contents
- Core algorithm modules:
  - `a_star_guidance.py`
  - `aimopso_operators.py`
  - `aimopso_runner.py`
  - `coordinates.py`
  - `cost_function.py`
  - `environments.py`
  - `pso_operators.py`
- Minimal dependency file: `requirements.txt`
- Reviewer evidence:
  - `assets/reviewer/` (example figures)
  - `results/demo/` (sample CSV output)

## Quick Start
```bash
pip install -r requirements.txt
python aimopso_runner.py --scene 1 --seed 42 --max-it 30 --n-pop 80 --n-rep 40
```

## Expected Output
After running the command above:
- A new demo CSV file is generated at:
  - `results/demo/latest_run_scene1.csv`
- Existing reviewer materials are available at:
  - `assets/reviewer/friedman_avg_ranks_smoke.png`
  - `assets/reviewer/friedman_avg_ranks_fair_pilot.png`

## Note on Full Release
Complete large-scale benchmark pipelines, full analysis scripts, and full experiment datasets will be released after paper acceptance.
