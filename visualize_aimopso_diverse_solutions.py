#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pymoo.config import Config
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

Config.warnings['not_compiled'] = False

try:
    from aimopso_runner import run_aimopso
    from plotting_matlab_exact_final2 import plot_and_save_paper_figures
    from environments import (
        create_scene1_model,
        create_scene2_model,
        create_scene3_model,
        create_scene4_model,
    )
    from algorithm_cache_manager import AlgorithmCacheManager
    from feasibility import is_feasible_cost
except ImportError as e:
    print(f"[ERROR] Failed to import project modules: {e}")
    sys.exit(1)


def derive_so_n_gen(max_fe, pop_size):
    return max(1, int(max_fe // pop_size))


def derive_core_max_it(max_fe, pop_size):
    if max_fe <= pop_size:
        return 1
    return max(1, int((max_fe - pop_size) // (2 * pop_size)))


def create_model(scene_id):
    creators = {
        1: create_scene1_model,
        2: create_scene2_model,
        3: create_scene3_model,
        4: create_scene4_model,
    }
    if scene_id not in creators:
        raise ValueError(f"Unknown scene: {scene_id}")
    model = creators[scene_id]()
    model['terrain_interpolator'] = RegularGridInterpolator(
        (np.arange(model['map_range'][1]), np.arange(model['map_range'][0])),
        model['H'],
        bounds_error=False,
        fill_value=0,
    )
    return model


def get_unified_absolute_path(waypoints_middle_only, model):
    start_pos, end_pos = model['start'], model['end']
    path_nodes_relative = np.vstack([start_pos, waypoints_middle_only, end_pos])

    ground_z = np.zeros(len(path_nodes_relative))
    for i in range(len(path_nodes_relative)):
        y_idx = int(np.round(path_nodes_relative[i, 1]))
        x_idx = int(np.round(path_nodes_relative[i, 0]))
        y_idx = np.clip(y_idx, 0, model['map_range'][1] - 1)
        x_idx = np.clip(x_idx, 0, model['map_range'][0] - 1)
        ground_z[i] = model['H'][y_idx, x_idx]

    path_nodes_absolute = path_nodes_relative.copy()
    path_nodes_absolute[:, 2] += ground_z
    return path_nodes_absolute


def plot_aimopso_with_custom_color(paths_absolute, path_labels, model, save_dir, scene_name, dpi=300):
    plot_and_save_paper_figures(
        paths_absolute=paths_absolute,
        path_labels=path_labels,
        model=model,
        save_dir=save_dir,
        scene_name=scene_name,
        dpi=dpi,
    )


def build_representative_results(paths_aimopso, rep_feasible):
    if not rep_feasible:
        return [], {}, []

    all_costs = [p['Cost'] for p in rep_feasible]
    feasible_costs = []
    feasible_indices = []
    for i, cost in enumerate(all_costs):
        if is_feasible_cost(cost):
            feasible_costs.append(np.asarray(cost, dtype=float))
            feasible_indices.append(i)

    if not feasible_costs:
        return [], {}, []

    feasible_costs_array = np.array(feasible_costs)
    nd_indices = NonDominatedSorting().do(feasible_costs_array, only_non_dominated_front=True)
    pareto_costs = [feasible_costs[i] for i in nd_indices]

    if paths_aimopso and len(paths_aimopso) == len(all_costs):
        feasible_paths = [paths_aimopso[i] for i in feasible_indices]
        pareto_paths = [feasible_paths[i] for i in nd_indices]
    else:
        pareto_paths = []

    final_results = {}
    if pareto_costs:
        pareto_costs_array = np.array(pareto_costs)
        total_costs = np.sum(pareto_costs_array, axis=1)
        excellent_indices = {
            'A*IMOPSO (Best Total)': int(np.argmin(total_costs)),
            'A*IMOPSO (Shortest)': int(np.argmin(pareto_costs_array[:, 0])),
            'A*IMOPSO (Safest)': int(np.argmin(pareto_costs_array[:, 1])),
            'A*IMOPSO (Lowest)': int(np.argmin(pareto_costs_array[:, 2])),
            'A*IMOPSO (Smoothest)': int(np.argmin(pareto_costs_array[:, 3])),
        }
        for name, idx in excellent_indices.items():
            final_results[name] = pareto_costs_array[idx]

    return pareto_paths, final_results, pareto_costs


def main():
    scene_to_run = int(os.getenv('SCENE_TO_RUN', '4'))
    seed = int(os.getenv('SEED', '42'))
    use_cache = os.getenv('USE_CACHE', '0').lower() in ('1', 'true', 'yes', 'y')
    clear_cache = os.getenv('CLEAR_CACHE', '0').lower() in ('1', 'true', 'yes', 'y')
    experiment_group = None
    save_dir = os.getenv('SAVE_DIR', 'aimopso_diverse_solutions_fair30')
    dpi = int(os.getenv('DPI', '300'))
    pop_size = int(os.getenv('POP_SIZE', '120'))
    n_rep = int(os.getenv('N_REP', '50'))
    max_fe = int(os.getenv('MAX_FE', '60120'))
    so_n_gen = derive_so_n_gen(max_fe, pop_size)
    max_it = derive_core_max_it(max_fe, pop_size)

    print('=' * 80)
    print('A*IMOPSO Diverse Solution Visualization')
    print('=' * 80)
    print(
        f'scene={scene_to_run}, seed={seed}, save_dir={save_dir}, '
        f'pop={pop_size}, max_fe={max_fe}, so_n_gen={so_n_gen}, '
        f'aimopso_max_it={max_it}, n_rep={n_rep}, use_cache={use_cache}'
    )

    cache_mgr = AlgorithmCacheManager()
    if clear_cache:
        print('[cache] clearing A*IMOPSO cache')
        cache_mgr.clear_cache(algorithm_name='A*IMOPSO')
        return

    common_params = {
        'pop_size': pop_size,
        'n_rep': n_rep,
        'max_fe': max_fe,
        'n_gen': so_n_gen,
        'max_it': max_it,
        'seed': seed,
    }

    model = create_model(scene_to_run)
    print(f"loaded scene: {model.get('name', 'unknown')}")

    algorithm_name = 'A*IMOPSO'
    scene_id = scene_to_run
    paths_aimopso = None
    rep_feasible = None

    if use_cache:
        cached_result = cache_mgr.load_result(algorithm_name, scene_id, common_params, experiment_group)
        if cached_result:
            print('[cache] loaded A*IMOPSO result')
            paths_aimopso = cached_result['pareto_paths']
            pareto_costs = cached_result['pareto_costs']
            rep_feasible = [{'Cost': cost} for cost in pareto_costs]
            execution_time = cached_result['time']
            print(f'[cache] done in {execution_time:.2f}s')

    if paths_aimopso is None or rep_feasible is None:
        print('[run] running A*IMOPSO ...')
        start_time = time.time()
        paths_aimopso, rep_feasible, _ = run_aimopso(
            model,
            seed=seed,
            mode='full',
            n_pop_override=pop_size,
            n_rep_override=n_rep,
            max_it_override=max_it,
        )
        execution_time = time.time() - start_time
        print(f'[run] done in {execution_time:.2f}s')

        if use_cache and rep_feasible:
            cache_data = {
                'pareto_paths': paths_aimopso,
                'pareto_costs': [p['Cost'] for p in rep_feasible],
                'time': execution_time,
            }
            cache_mgr.save_result(algorithm_name, scene_id, common_params, cache_data, experiment_group)
            print('[cache] saved A*IMOPSO result')

    pareto_paths, final_results, pareto_costs = build_representative_results(paths_aimopso, rep_feasible)

    if final_results:
        print('-' * 80)
        print(f"{'Solution Type':<28} | {'J1':<12} | {'J2':<12} | {'J3':<12} | {'J4':<12}")
        print('-' * 80)
        for name, costs in sorted(final_results.items()):
            print(f"{name:<28} | {costs[0]:<12.4f} | {costs[1]:<12.4f} | {costs[2]:<12.4f} | {costs[3]:<12.4f}")
        print('-' * 80)
    else:
        print('[warn] no feasible A*IMOPSO result found')

    if pareto_paths:
        best_total_name = 'A*IMOPSO (Best Total)'
        best_path = None
        if best_total_name in final_results:
            target = final_results[best_total_name]
            for idx, costs in enumerate(pareto_costs):
                if np.allclose(costs, target):
                    best_path = pareto_paths[idx]
                    break
        if best_path is None:
            best_path = pareto_paths[0]

        scene_name = f'scene_{scene_to_run}'
        paths_absolute = [get_unified_absolute_path(best_path, model)]
        plot_aimopso_with_custom_color(
            paths_absolute=paths_absolute,
            path_labels=['A*IMOPSO'],
            model=model,
            save_dir=save_dir,
            scene_name=scene_name,
            dpi=dpi,
        )
        print(f'[done] figures saved to {save_dir}')
    else:
        print('[skip] no path available for plotting')

    print('[cache] summary:')
    cache_mgr.list_cache()


if __name__ == '__main__':
    main()
