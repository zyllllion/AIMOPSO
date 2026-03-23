import numpy as np
def dominates(cost1, cost2):
    """
    cost1 是否支配 cost2
    """
    if np.any(np.isinf(cost1)):
        return False
    if np.any(np.isinf(cost2)) and not np.any(np.isinf(cost1)):
        return True
    return np.all(cost1 <= cost2) and np.any(cost1 < cost2)
def determine_domination(pop):
    """
    MATLAB 样式双层支配判断（包含最后元素单独处理）
    含 Inf 的个体初步标记为支配；后续若全部 Inf 会用 fallback 重新挑选一个。
    """
    n = len(pop)
    for p in pop:
        p['IsDominated'] = False
        if np.any(np.isinf(p['Cost'])):
            p['IsDominated'] = True
    # 前 n-1
    for i in range(n - 1):
        if pop[i]['IsDominated']:
            continue
        for j in range(i + 1, n):
            if dominates(pop[i]['Cost'], pop[j]['Cost']):
                pop[j]['IsDominated'] = True
            if dominates(pop[j]['Cost'], pop[i]['Cost']):
                pop[i]['IsDominated'] = True
    # 最后一个
    if n > 0:
        if np.any(np.isinf(pop[-1]['Cost'])):
            pop[-1]['IsDominated'] = True
        else:
            for j in range(n - 1):
                if dominates(pop[-1]['Cost'], pop[j]['Cost']):
                    pop[j]['IsDominated'] = True
                if dominates(pop[j]['Cost'], pop[-1]['Cost']):
                    pop[-1]['IsDominated'] = True
                    break
    return pop
def ensure_non_empty_rep(all_solutions):
    """
    如果所有解都被标记为支配导致 rep 为空，挑一个"最不坏"解：
      1. 按 Inf 个数升序
      2. 再按 替换 Inf->1e9 后的总和 升序
    返回新的 rep 列表（保证至少一个）
    """
    rep = [p for p in all_solutions if not p['IsDominated']]
    if rep:
        return rep
    # 计算排序键
    def key_fn(p):
        cost = p['Cost']
        inf_mask = np.isinf(cost)
        inf_count = np.count_nonzero(inf_mask)
        surrogate_sum = np.sum(np.where(inf_mask, 1e9, cost))
        return (inf_count, surrogate_sum)
    best = min(all_solutions, key=key_fn)
    best['IsDominated'] = False  # 解除 dominated 标记
    return [best]
def create_grid(rep, n_grid, alpha):
    if not rep:
        return []
    costs = np.array([p['Cost'] for p in rep]).T  # (n_obj, n_rep)
    cmin = np.min(costs, axis=1)
    cmax = np.max(costs, axis=1)
    dc = cmax - cmin
    cmin -= alpha * dc
    cmax += alpha * dc
    grid = []
    for j in range(costs.shape[0]):
        cj = np.linspace(cmin[j], cmax[j], n_grid + 1)
        lb = np.concatenate(([-np.inf], cj))
        ub = np.concatenate((cj, [np.inf]))
        grid.append({'LB': lb, 'UB': ub})
    return grid
def find_grid_index(p, grid):
    if not grid:
        p['GridSubIndex'] = np.array([], dtype=int)
        p['GridIndex'] = 0
        return p
    n_obj = len(p['Cost'])
    n_grid = len(grid[0]['LB']) - 2
    sub_idx = np.zeros(n_obj, dtype=int)
    for j in range(n_obj):
        res = np.where(p['Cost'][j] < grid[j]['UB'])[0]
        idx = res[0] if len(res) else n_grid + 1
        sub_idx[j] = idx
    p['GridSubIndex'] = sub_idx
    gi = sub_idx[0] - 1
    for j in range(1, n_obj):
        gi = n_grid * gi + (sub_idx[j] - 1)
    p['GridIndex'] = gi
    return p
def roulette_wheel_selection(P):
    """
    稳健轮盘赌：
      - 若 P 为空 -> raise
      - 若 sum(P)==0 -> 均匀分布
      - 若浮点误差使 r > C[-1] -> 返回最后一个
    """
    P = np.asarray(P, dtype=float)
    if P.size == 0:
        raise ValueError("roulette_wheel_selection 收到空概率向量。")
    s = np.sum(P)
    if not np.isfinite(s) or s <= 0:
        P = np.full_like(P, 1.0 / P.size)
    else:
        P = P / s
    C = np.cumsum(P)
    r = np.random.rand()
    idxs = np.where(r <= C)[0]
    if idxs.size == 0:
        return P.size - 1
    return idxs[0]
def select_leader(rep, beta):
    """
    选择仓库中的领导者（多样性）。
    加稳健防护：若 rep 空 -> raise。
    概率用稳定指数防止下溢。
    """
    if not rep:
        raise ValueError("select_leader: rep 为空，无法选择领导者。")
    grid_indices = [p['GridIndex'] for p in rep]
    unique_indices, counts = np.unique(grid_indices, return_counts=True)
    # 稳定指数
    raw = -beta * counts.astype(float)
    raw -= np.max(raw)  # 防下溢
    weights = np.exp(raw)
    if np.all(weights == 0):
        weights = np.ones_like(weights)
    probs = weights / np.sum(weights)
    cell_idx = roulette_wheel_selection(probs)
    selected_grid = unique_indices[cell_idx]
    members = [i for i, idx in enumerate(grid_indices) if idx == selected_grid]
    return rep[np.random.randint(0, len(members))]
def delete_one_rep_member(rep, gamma):
    grid_indices = [p['GridIndex'] for p in rep]
    unique_indices, counts = np.unique(grid_indices, return_counts=True)
    raw = gamma * counts.astype(float)
    raw -= np.max(raw)
    weights = np.exp(raw)
    if np.all(weights == 0):
        weights = np.ones_like(weights)
    probs = weights / np.sum(weights)
    cell_idx = roulette_wheel_selection(probs)
    selected_grid = unique_indices[cell_idx]
    members = [i for i, idx in enumerate(grid_indices) if idx == selected_grid]
    idx_to_delete = members[np.random.randint(0, len(members))]
    rep.pop(idx_to_delete)
    return rep
def mutate(particle, delta, var_max, var_min):
    """
    合理化变异
    """
    pos = particle['Position']
    pbest_pos = particle['Best']['Position']
    n_var = len(pos['r'][0])
    beta = np.tanh(delta * 1.0)
    new_pos = {}
    for key in pos.keys():
        noise = np.random.randn(1, n_var) * pbest_pos[key] * beta
        new_val = pos[key] + noise
        new_val = np.clip(new_val, var_min[key], var_max[key])
        new_pos[key] = new_val
    return new_pos

def choose_deterministic_rep(rep):
    """
    字典序挑选确定代表（如果 rep 空返回 None）
    """
    if not rep:
        return None
    rep_sorted = sorted(rep, key=lambda p: tuple(p['Cost']))
    return rep_sorted[0]
