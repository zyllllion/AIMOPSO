import numpy as np
import random
from scipy.interpolate import RegularGridInterpolator

from coordinates import spherical_to_cartesian, cartesian_to_spherical
from a_star_guidance import get_a_star_guidance_path
from cost_function import calculate_cost
from pso_operators import dominates
from aimopso_operators import (
    get_new_repository,
    polynomial_mutation,
    select_leader_by_tournament
)

FEASIBILITY_THRESHOLD = np.inf
INFEASIBLE_PENALTY = np.inf


def _is_feasible_cost(cost_vec):
    """Unified feasibility check function"""
    return (cost_vec is not None and np.all(np.isfinite(cost_vec)))


def _filter_feasible(pop):
    return [p for p in pop if _is_feasible_cost(p.get('Cost', None))]


def run_aimopso(model, seed=None, mode='stats', a_star_weight=None, use_a_star_init=True, use_dual_leader=True):
    """
    A*IMOPSO algorithm.
    
    Args:
        model: Environment model
        seed: Random seed
        mode: Run mode ('stats' or 'visual')
        a_star_weight: A* path influence weight on population initialization (0-1)
            - None: Auto-adjust based on scene complexity
            - 0.0: Fully random initialization
            - 1.0: Fully use A* path initialization
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    if a_star_weight is None:
        a_star_weight = 0.8
    else:
        a_star_weight = np.clip(float(a_star_weight), 0.0, 1.0)

    n_var = model['n']
    var_size = (1, n_var)
    var_max, var_min = {}, {}
    var_max['r'] = 3 * np.linalg.norm(model['start'] - model['end']) / n_var
    var_min['r'] = var_max['r'] / 9
    angle_range = np.pi / 4
    var_min['psi'], var_max['psi'] = -angle_range, angle_range
    var_min['phi'], var_max['phi'] = -angle_range, angle_range

    alpha_vel = 0.5
    vel_max, vel_min = {}, {}
    for key in ['r', 'psi', 'phi']:
        vel_max[key] = alpha_vel * (var_max[key] - var_min[key])
        vel_min[key] = -vel_max[key]

    if 'terrain_interpolator' not in model:
        model['terrain_interpolator'] = RegularGridInterpolator(
            (np.arange(model['map_range'][1]), np.arange(model['map_range'][0])),
            model['H'], bounds_error=False, fill_value=0
        )
    terrain_interpolator = model['terrain_interpolator']

    max_it, n_pop, n_rep = 500, 100, 50
    w, wdamp, c1, c2 = 1.0, 0.98, 1.5, 1.5
    prob_attack_force = 0.7

    if use_a_star_init:
        print("    -> Attempting to generate A* guidance path...")
        guidance_path_xy = get_a_star_guidance_path(model)

        if guidance_path_xy is not None:
            print("    -> A* guidance successful! Initializing population based on guidance path...")
            safe_avg_altitude = (model['zmin'] + model['zmax']) / 2
            
            guidance_path_z_rel = np.full(len(guidance_path_xy), safe_avg_altitude)
            
            guidance_path_xyz_rel = np.hstack([guidance_path_xy, guidance_path_z_rel.reshape(-1, 1)])

            template_pos = cartesian_to_spherical(guidance_path_xyz_rel, model)
        else:
            print("    -> A* guidance failed. Falling back to random initialization.")
            template_pos = None
    else:
        print("    -> A* guidance disabled, using random initialization.")
        template_pos = None

    particles = []
    n_pure_astar = max(1, int(n_pop * 0.05))
    n_low_noise = int(n_pop * 0.50)
    n_normal_noise = n_pop - n_pure_astar - n_low_noise
    
    for i in range(n_pop):
        pos = {}
        if template_pos is not None:
            if i < n_pure_astar:
                pos['r'] = template_pos['r'].copy()
                pos['psi'] = template_pos['psi'].copy()
                pos['phi'] = template_pos['phi'].copy()
            elif i < n_pure_astar + n_low_noise:
                noise_r = np.random.uniform(-0.05, 0.05, var_size) * (var_max['r'] - var_min['r'])
                noise_psi = np.random.uniform(-0.1, 0.1, var_size) * (var_max['psi'] - var_min['psi'])
                noise_phi = np.random.uniform(-0.1, 0.1, var_size) * (var_max['phi'] - var_min['phi'])
                pos['r'] = template_pos['r'] + noise_r
                pos['psi'] = template_pos['psi'] + noise_psi
                pos['phi'] = template_pos['phi'] + noise_phi
            else:
                noise_r = np.random.uniform(-0.1, 0.1, var_size) * (var_max['r'] - var_min['r'])
                noise_psi = np.random.uniform(-0.2, 0.2, var_size) * (var_max['psi'] - var_min['psi'])
                noise_phi = np.random.uniform(-0.2, 0.2, var_size) * (var_max['phi'] - var_min['phi'])
                pos['r'] = template_pos['r'] + noise_r
                pos['psi'] = template_pos['psi'] + noise_psi
                pos['phi'] = template_pos['phi'] + noise_phi
        else:
            pos['r'] = np.random.uniform(var_min['r'], var_max['r'], var_size)
            pos['psi'] = np.random.uniform(var_min['psi'], var_max['psi'], var_size)
            pos['phi'] = np.random.uniform(var_min['phi'], var_max['phi'], var_size)
    
        for k in pos.keys():
            pos[k] = np.clip(pos[k], var_min[k], var_max[k])
    
        vel = {k: np.zeros(var_size) for k in pos.keys()}
        cost = calculate_cost(spherical_to_cartesian(pos, model), model, terrain_interpolator, var_min)
        particles.append({
            'Position': pos, 'Velocity': vel, 'Cost': cost,
            'Best': {'Position': {k: v.copy() for k, v in pos.items()}, 'Cost': None if cost is None else cost.copy()},
            'CrowdingDistance': 0
        })

    init_pool = _filter_feasible(particles)
    rep = get_new_repository(init_pool, n_rep) if len(init_pool) > 0 else []
    
    if template_pos is not None:
        print(f"    [A*IMOPSO] A* guided initialization complete: {n_pop} particles, feasible: {len(init_pool)}")
        print(f"               ├─ 5% pure A*: {n_pure_astar}")
        print(f"               ├─ 50% low noise (5-10%): {n_low_noise}")
        print(f"               └─ 45% normal noise (10-20%): {n_normal_noise}")

    for it in range(max_it):
        costs = np.array([p['Cost'] for p in rep])
        global_best_leader = rep[np.argmin(np.sum(costs, axis=1))]
        for i in range(n_pop):
            p = particles[i]
            if not rep: break
            if use_dual_leader:
                leader = global_best_leader if np.random.rand() < prob_attack_force else select_leader_by_tournament(rep)
            else:
                leader = select_leader_by_tournament(rep)
            for key in ['r', 'psi', 'phi']:
                r1, r2 = np.random.rand(*var_size), np.random.rand(*var_size)
                p['Velocity'][key] = (w * p['Velocity'][key] + c1 * r1 * (
                            p['Best']['Position'][key] - p['Position'][key]) + c2 * r2 * (
                                                  leader['Position'][key] - p['Position'][key]))
                p['Velocity'][key] = np.clip(p['Velocity'][key], vel_min[key], vel_max[key])
                p['Position'][key] += p['Velocity'][key]
                out_mask = (p['Position'][key] < var_min[key]) | (p['Position'][key] > var_max[key])
                p['Velocity'][key][out_mask] *= -1
                p['Position'][key] = np.clip(p['Position'][key], var_min[key], var_max[key])
            p['Cost'] = calculate_cost(spherical_to_cartesian(p['Position'], model), model, terrain_interpolator,
                                       var_min)
            if p['Best']['Cost'] is None or dominates(p['Cost'], p['Best']['Cost']) or (
                    not dominates(p['Best']['Cost'], p['Cost']) and np.random.rand() < 0.5):
                p['Best']['Position'] = {k: v.copy() for k, v in p['Position'].items()}
                p['Best']['Cost'] = None if p['Cost'] is None else p['Cost'].copy()
        mutated_particles = []
        for p in particles:
            new_pos = polynomial_mutation(p, var_max, var_min, prob_mut=0.2)
            mutated_particles.append({
                'Position': new_pos, 'Velocity': p['Velocity'],
                'Cost': calculate_cost(spherical_to_cartesian(new_pos, model), model, terrain_interpolator, var_min),
                'Best': {'Position': new_pos, 'Cost': np.inf}, 'CrowdingDistance': 0
            })
        candidate_pool = _filter_feasible(rep + particles + mutated_particles)
        rep = get_new_repository(candidate_pool, n_rep) if len(candidate_pool) > 0 else []
        
        w *= wdamp

    if not rep:
        if mode == 'stats': return np.empty((0, 4))
        return (None, None) if mode in ['visual', 'full'] else np.empty((0, 4))

    rep_feasible = _filter_feasible(rep)
    if not rep_feasible:
        if mode == 'stats': return np.empty((0, 4))
        return (None, None) if mode in ['visual', 'full'] else np.empty((0, 4))

    model_for_plot = {
        'H': model['H'], 'threats': model.get('threats', np.array([])),
        'map_range': [model['map_range'][0], model['map_range'][1]]
    }

    if mode == 'stats':
        return np.array([p['Cost'] for p in rep_feasible])
    elif mode == 'visual':
        best_solution = rep_feasible[np.argmin(np.sum(np.array([p['Cost'] for p in rep_feasible]), axis=1))]
        final_cart = spherical_to_cartesian(best_solution['Position'], model)
        path_points_waypoints_only = np.column_stack([final_cart['x'], final_cart['y'], final_cart['z']])
        return path_points_waypoints_only, model_for_plot
    elif mode == 'full':
        all_paths = []
        for solution in rep_feasible:
            final_cart = spherical_to_cartesian(solution['Position'], model)
            path_points_waypoints_only = np.column_stack([final_cart['x'], final_cart['y'], final_cart['z']])
            all_paths.append(path_points_waypoints_only)
        return all_paths, rep_feasible, model_for_plot
    else:
        raise ValueError(f"Unknown mode: '{mode}'. Use 'stats', 'visual', or 'full'.")
