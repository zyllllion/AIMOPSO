import numpy as np
import random
from scipy.interpolate import RegularGridInterpolator

# --- 导入模块 ---
# 【重要】确保这里没有 from environment... import create_model
from coordinates import spherical_to_cartesian, cartesian_to_spherical
from a_star_guidance import get_a_star_guidance_path
# from local_search import apply_local_search_to_repository  # 已删除局部搜索功能
from cost_function import calculate_cost
from pso_operators import dominates
from aimopso_operators import (
    get_new_repository,
    polynomial_mutation,
    select_leader_by_tournament
)

# 🔧 统一的约束处理常量（与MATLAB NMOPSO-main保持完全一致）
FEASIBILITY_THRESHOLD = np.inf   # 可行性阈值（对应MATLAB的J_inf = inf）
INFEASIBLE_PENALTY = np.inf      # 不可行解惩罚值（对应MATLAB的inf）


def _is_feasible_cost(cost_vec):
    """统一的可行性检查函数（对应MATLAB的all(x < inf)）"""
    return (cost_vec is not None and np.all(np.isfinite(cost_vec)))


def _filter_feasible(pop):
    return [p for p in pop if _is_feasible_cost(p.get('Cost', None))]


# def _get_dynamic_prob_attack_force(iteration, max_iterations):
#     """
#     根据迭代进度动态调整prob_attack_force（探索-开发平衡）
#     
#     参数:
#         iteration: 当前迭代次数 (0-based)
#         max_iterations: 最大迭代次数
#         
#     返回:
#         float: 动态计算的prob_attack_force (0.0-1.0)
#         
#     策略说明:
#         - 早期 (0-30%): 高概率探索 (0.7-0.8)，增加种群多样性
#         - 中期 (30-70%): 逐渐降低 (0.8-0.5)，平衡探索与开发
#         - 后期 (70-100%): 低概率探索 (0.5-0.3)，专注于精细开发
#     """
#     # 计算归一化进度 [0, 1]
#     progress = iteration / max_iterations
#     
#     # 方案1: 线性递减（简单直接）
#     # start_prob, end_prob = 0.8, 0.3
#     # prob = start_prob - (start_prob - end_prob) * progress
#     
#     # 方案2: 分段线性（更精细控制）
#     if progress < 0.3:  # 早期：保持高探索
#         prob = 0.8
#     elif progress < 0.7:  # 中期：线性递减
#         prob = 0.8 - 0.3 * ((progress - 0.3) / 0.4)  # 0.8 → 0.5
#     else:  # 后期：继续降低
#         prob = 0.5 - 0.2 * ((progress - 0.7) / 0.3)  # 0.5 → 0.3
#     
#     # 方案3: 余弦退火（平滑过渡）- 备选
#     # start_prob, end_prob = 0.8, 0.3
#     # prob = end_prob + 0.5 * (start_prob - end_prob) * (1 + np.cos(np.pi * progress))
#     
#     # 确保在有效范围内
#     prob = np.clip(prob, 0.2, 0.9)
#     
#     # 每50次迭代输出一次（避免刷屏）
#     if iteration % 50 == 0:
#         print(f"  [迭代 {iteration}/{max_iterations}] prob_attack_force = {prob:.3f} (进度: {progress*100:.1f}%)")
#     
#     return prob



# ======================= 核心修复: 修改函数签名，接收 model 参数 =======================
def run_aimopso(
    model,
    seed=None,
    mode='stats',
    a_star_weight=None,
    use_a_star_init=True,
    use_dual_leader=True,
    max_it_override=None,
    n_pop_override=None,
    n_rep_override=None,
):
    """
    A*IMOPSO 算法。
    
    参数:
        model: 环境模型
        seed: 随机种子
        mode: 运行模式 ('stats' 或 'visual')
        a_star_weight: A*路径对种群初始化的影响权重 (0-1)
            - None: 自动根据场景复杂度调整
            - 0.0: 完全随机初始化
            - 1.0: 完全使用A*路径初始化
        max_it_override / n_pop_override / n_rep_override:
            仅用于实验协议控制，默认 None 时保持算法原始配置不变。
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # 使用固定A*权重
    if a_star_weight is None:
        a_star_weight = 0.8  # 固定权重，与历史基线一致
    else:
        # 确保权重在有效范围内
        a_star_weight = np.clip(float(a_star_weight), 0.0, 1.0)

    # 不再自己创建模型，直接使用传入的 model 对象
    # 1. 初始化参数 (直接使用传入的 model)
    n_var = model['n']
    var_size = (1, n_var)
    var_max, var_min = {}, {}
    var_max['r'] = 3 * np.linalg.norm(model['start'] - model['end']) / n_var
    var_min['r'] = var_max['r'] / 9
    # 增强探索 1: 将角度搜索范围扩大到 [-pi, pi]，允许向任何方向探索
    # 设置为 pi/2 (180度半球)，为算法提供足够的局部机动性，同时符合物理直觉
    angle_range = np.pi / 4
    var_min['psi'], var_max['psi'] = -angle_range, angle_range
    var_min['phi'], var_max['phi'] = -angle_range, angle_range

    alpha_vel = 0.5
    vel_max, vel_min = {}, {}
    for key in ['r', 'psi', 'phi']:
        vel_max[key] = alpha_vel * (var_max[key] - var_min[key])
        vel_min[key] = -vel_max[key]

    # 确保地形插值器存在
    if 'terrain_interpolator' not in model:
        model['terrain_interpolator'] = RegularGridInterpolator(
            (np.arange(model['map_range'][1]), np.arange(model['map_range'][0])),
            model['H'], bounds_error=False, fill_value=0
        )
    terrain_interpolator = model['terrain_interpolator']

    max_it, n_pop, n_rep = 500, 100, 50
    if max_it_override is not None:
        max_it = int(max_it_override)
    if n_pop_override is not None:
        n_pop = int(n_pop_override)
    if n_rep_override is not None:
        n_rep = int(n_rep_override)
    # ⚠️ PSO参数设置（与标准PSO保持一致以确保公平对比）
    w, wdamp, c1, c2 = 1.0, 0.98, 1.5, 1.5
    # ✅ 最优prob_attack_force（经实验验证）
    prob_attack_force = 0.7  # 经过消融实验验证的最优值

    # (后续代码保持不变, 因为它们都依赖于 model 对象)
    # ... (初始化种群) ...
    if use_a_star_init:
        print("    -> 尝试使用 A* 算法生成引导路径...")
        guidance_path_xy = get_a_star_guidance_path(model)

        if guidance_path_xy is not None:
            print("    -> A* 引导成功！正在基于引导路径初始化种群...")
            # --- 策略一：使用A*引导路径创建相对高度的三维路径 ---
            # 1. 定义一个安全的平均飞行高度（相对高度，离地高度）
            safe_avg_altitude = (model['zmin'] + model['zmax']) / 2
            
            # 2. 创建一条使用相对高度的三维引导路径
            # 注意：这里z使用相对高度（离地高度），而不是绝对高度（海平面高度）
            # 这样才能与cartesian_to_spherical函数的预期输入一致
            guidance_path_z_rel = np.full(len(guidance_path_xy), safe_avg_altitude)
            
            # 3. 拼接成完整的三维相对坐标航路点
            guidance_path_xyz_rel = np.hstack([guidance_path_xy, guidance_path_z_rel.reshape(-1, 1)])

            # 将相对高度路径转换为球坐标模板
            template_pos = cartesian_to_spherical(guidance_path_xyz_rel, model)
        else:
            print("    -> A* 引导失败。退回至随机初始化。")
            template_pos = None
    else:
        print("    -> 已禁用 A* 引导，执行随机初始化。")
        template_pos = None

    # ============ 【最终最优配置】分层噪声初始化策略 ============
    # 经过5个消融实验（A-E）验证，当前配置已达最优平衡点！
    # 核心价值：
    # 1. 保证算法鲁棒性（实验B证明：原始初始化+网格150会失败）
    # 2. 实现最优性能（J1=0.0664，超越所有算法）
    # 3. 平衡探索与开发（J4=0.0838，接近PSO的0.0807）
    # 
    # 消融实验总结：
    # - 实验C（网格200）：J1和J4都变差，证明网格150最优
    # - 实验D（15%纯+65%超低）：探索不足，J1=0.0780, J4=0.0940
    # - 实验E（10%纯+50%低+40%正常）：仍不如当前配置
    # 
    # 最优配置（经验证）：
    # - 5%：纯A*路径（零噪声）-> 保证鲁棒性
    # - 50%：低噪声A*路径（5-10%）-> 局部精细搜索
    # - 45%：正常噪声A*路径（10-20%）-> 全局多样性探索
    particles = []
    n_pure_astar = max(1, int(n_pop * 0.05))       # 5%纯A*路径
    n_low_noise = int(n_pop * 0.50)                # 50%低噪声
    n_normal_noise = n_pop - n_pure_astar - n_low_noise  # 剩余45%正常噪声
    
    for i in range(n_pop):
        pos = {}
        if template_pos is not None:
            if i < n_pure_astar:
                # 【第1层】纯A*路径（零噪声）- 保证最优起点
                pos['r'] = template_pos['r'].copy()
                pos['psi'] = template_pos['psi'].copy()
                pos['phi'] = template_pos['phi'].copy()
            elif i < n_pure_astar + n_low_noise:
                # 【第2层】低噪声A*路径（5-10%）- 局部搜索
                noise_r = np.random.uniform(-0.05, 0.05, var_size) * (var_max['r'] - var_min['r'])
                noise_psi = np.random.uniform(-0.1, 0.1, var_size) * (var_max['psi'] - var_min['psi'])
                noise_phi = np.random.uniform(-0.1, 0.1, var_size) * (var_max['phi'] - var_min['phi'])
                pos['r'] = template_pos['r'] + noise_r
                pos['psi'] = template_pos['psi'] + noise_psi
                pos['phi'] = template_pos['phi'] + noise_phi
            else:
                # 【第3层】正常噪声A*路径（10-20%）- 全局探索
                noise_r = np.random.uniform(-0.1, 0.1, var_size) * (var_max['r'] - var_min['r'])
                noise_psi = np.random.uniform(-0.2, 0.2, var_size) * (var_max['psi'] - var_min['psi'])
                noise_phi = np.random.uniform(-0.2, 0.2, var_size) * (var_max['phi'] - var_min['phi'])
                pos['r'] = template_pos['r'] + noise_r
                pos['psi'] = template_pos['psi'] + noise_psi
                pos['phi'] = template_pos['phi'] + noise_phi
        else:
            # 完全随机生成（A*失败时的后备方案）
            pos['r'] = np.random.uniform(var_min['r'], var_max['r'], var_size)
            pos['psi'] = np.random.uniform(var_min['psi'], var_max['psi'], var_size)
            pos['phi'] = np.random.uniform(var_min['phi'], var_max['phi'], var_size)
    
        # 确保生成的粒子仍在范围内
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
    
    # 打印初始化统计信息
    if template_pos is not None:
        print(f"    [A*IMOPSO] A*引导初始化完成: {n_pop} 个粒子, 可行: {len(init_pool)} 个")
        print(f"               ├─ 5% 纯A*路径: {n_pure_astar} 个")
        print(f"               ├─ 50% 低噪声(5-10%): {n_low_noise} 个")
        print(f"               └─ 45% 正常噪声(10-20%): {n_normal_noise} 个")

    # --- 停滞检测变量（已注释，不再使用）---
    # stagnation_counter = 0
    # stagnation_threshold = 25  # 连续25代无变化则触发
    # last_rep_costs_str = ""
    # --------------------------------

    # 3. 主循环
    for it in range(max_it):
        # --- 停滞检测逻辑已注释（未被使用）---
        # 多项式变异固定以 P_mut=0.2 的概率每代执行，不依赖停滞检测
        # current_rep_costs = np.array([p['Cost'] for p in rep])
        # current_rep_costs_str = np.array2string(current_rep_costs)
        # if current_rep_costs_str == last_rep_costs_str and len(rep) > 0:
        #     stagnation_counter += 1
        # else:
        #     stagnation_counter = 0
        # last_rep_costs_str = current_rep_costs_str
        # --------------------------------

        if not rep: break
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
        
        # --- 局部搜索已删除（存在坐标转换误差问题）---
        # 依靠 A*引导初始化 + 80%全局搜索 + 多项式变异 即可达到优秀性能
        
        w *= wdamp

    # ... (返回部分) ...
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
        # 🔧 修复：只返回中间航路点，与其他算法保持一致
        path_points_waypoints_only = np.column_stack([final_cart['x'], final_cart['y'], final_cart['z']])
        return path_points_waypoints_only, model_for_plot
    elif mode == 'full':
        all_paths = []
        for solution in rep_feasible:
            final_cart = spherical_to_cartesian(solution['Position'], model)
            # 🔧 修复：只返回中间航路点，与其他算法保持一致
            path_points_waypoints_only = np.column_stack([final_cart['x'], final_cart['y'], final_cart['z']])
            all_paths.append(path_points_waypoints_only)
        # Return the full path set, the repository, and the model for plotting
        return all_paths, rep_feasible, model_for_plot
    else:
        raise ValueError(f"未知的模式: '{mode}'。请使用 'stats', 'visual', 或 'full'。")
