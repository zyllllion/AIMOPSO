 

# he_nmopso_operators.py
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# --------------------------------------------------
# 拥挤度距离计算
# --------------------------------------------------
def calculate_crowding_distance(F):
    n_points, n_obj = F.shape
    if n_points <= 2:
        return np.full(n_points, np.inf)

    crowding = np.zeros(n_points)

    for j in range(n_obj):
        sorted_indices = np.argsort(F[:, j])
        sorted_F = F[sorted_indices, j]

        # 边界点 = 无穷大
        crowding[sorted_indices[0]] = np.inf
        crowding[sorted_indices[-1]] = np.inf

        f_min, f_max = sorted_F[0], sorted_F[-1]
        norm = f_max - f_min
        if norm < 1e-12:  # 避免除以零
            continue

        for i in range(1, n_points - 1):
            crowding[sorted_indices[i]] += (sorted_F[i + 1] - sorted_F[i - 1]) / norm

    return np.nan_to_num(crowding, nan=0.0, posinf=np.inf)


# --------------------------------------------------
# 精英保留：更新外部存档
# --------------------------------------------------
def get_new_repository(combined_pop, rep_size):
    """
    输入：combined_pop (list of particles, 每个 particle 至少有 'Cost')
    输出：截断后的新存档 (list)
    """
    if not combined_pop:
        return []

    costs = np.array([p['Cost'] for p in combined_pop])

    nds = NonDominatedSorting()
    fronts = nds.do(costs, only_non_dominated_front=False)

    new_rep = []
    for front_indices in fronts:
        front_costs = costs[front_indices]
        crowding_distances = calculate_crowding_distance(front_costs)

        for i, idx in enumerate(front_indices):
            combined_pop[idx]['CrowdingDistance'] = crowding_distances[i]

        if len(new_rep) + len(front_indices) <= rep_size:
            for idx in front_indices:
                new_rep.append(combined_pop[idx])
        else:
            num_needed = rep_size - len(new_rep)
            last_front_particles = [combined_pop[idx] for idx in front_indices]
            sorted_last_front = sorted(
                last_front_particles, key=lambda p: p['CrowdingDistance'], reverse=True
            )
            new_rep.extend(sorted_last_front[:num_needed])
            break

    return new_rep


# --------------------------------------------------
# 锦标赛选择领导者
# --------------------------------------------------
def dominates(cost1, cost2):
    """判断 cost1 是否支配 cost2"""
    return np.all(cost1 <= cost2) and np.any(cost1 < cost2)


def select_leader_by_tournament(rep):
    if len(rep) == 0:
        return None
    if len(rep) == 1:
        return rep[0]

    i1, i2 = np.random.randint(len(rep)), np.random.randint(len(rep))
    p1, p2 = rep[i1], rep[i2]

    if dominates(p1['Cost'], p2['Cost']):
        return p1
    elif dominates(p2['Cost'], p1['Cost']):
        return p2
    elif p1.get('CrowdingDistance', 0) > p2.get('CrowdingDistance', 0):
        return p1
    elif p2.get('CrowdingDistance', 0) > p1.get('CrowdingDistance', 0):
        return p2
    else:
        return p1 if np.random.rand() < 0.5 else p2


# --------------------------------------------------
# 多项式变异
# --------------------------------------------------
def polynomial_mutation(particle, var_max, var_min, eta=20, prob_mut=0.2):
    new_pos = {k: v.copy() for k, v in particle['Position'].items()}

    for key in ['r', 'psi', 'phi']:
        for i in range(new_pos[key].shape[1]):
            if np.random.rand() <= prob_mut:
                y = new_pos[key][0, i]
                y_min, y_max = var_min[key], var_max[key]

                delta1 = (y - y_min) / (y_max - y_min)
                delta2 = (y_max - y) / (y_max - y_min)

                rand_val = np.random.rand()
                mut_pow = 1.0 / (eta + 1.0)

                if rand_val < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand_val + (1.0 - 2.0 * rand_val) * (xy ** (eta + 1.0))
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand_val) + 2.0 * (rand_val - 0.5) * (xy ** (eta + 1.0))
                    delta_q = 1.0 - val ** mut_pow

                y = y + delta_q * (y_max - y_min)
                y = np.clip(y, y_min, y_max)
                new_pos[key][0, i] = y

    return new_pos
