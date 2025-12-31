# A*IMOPSO Algorithm Source Code

## Overview

This repository contains the source code for the A*-Guided Heuristic Multi-Objective Particle Swarm Optimization (A*IMOPSO) algorithm. The current implementation is designed for path planning of Unmanned Aerial Vehicles (UAVs). However, with modifications, it can be applied to other optimization problems.

## Running the Code

To run the A*IMOPSO algorithm:

1. Download all the source files from this repository.
2. Install Python 3.8 or higher.
3. Install required dependencies:
   ```bash
   pip install numpy scipy matplotlib
   ```
4. Execute the main script by running:
   ```bash
   python visualize_aimopso_diverse_solutions.py
   ```

## File Structure

```
aimopso/
├── aimopso_operators.py                      # Core A*IMOPSO algorithm
├── a_star_guidance.py                        # A* guided initialization
├── pso_operators.py                          # PSO operators
├── cost_function.py                          # Multi-objective cost functions
├── environments.py                           # Environment modeling
├── coordinates.py                            # Coordinate transformations
├── aimopso_runner.py                         # Algorithm runner
├── visualize_aimopso_diverse_solutions.py    # Main visualization script
├── algorithm_cache_manager.py                # Cache manager
├── plotting_matlab_exact_final2.py           # Plotting utilities
└── README.md                                 # This file
```


