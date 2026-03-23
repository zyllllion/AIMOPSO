#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A*IMOPSO 多样化解集可视化脚本
- 专门用于可视化A*IMOPSO算法的多样化解集
- 生成高质量的3D和俯视图，用于论文展示
- 保存格式与对比可视化脚本一致
"""

import numpy as np
import time
import sys
from scipy.interpolate import RegularGridInterpolator

from pymoo.config import Config
Config.warnings['not_compiled'] = False

try:
    from aimopso_runner import run_aimopso
    from plotting_matlab_exact_final2 import plot_and_save_paper_figures
    from environments import create_scene1_model, create_scene2_model, create_scene3_model,create_scene4_model
    from cost_function import calculate_cost
    from algorithm_cache_manager import AlgorithmCacheManager
except ImportError as e:
    print(f"【错误】: 导入自定义模块失败 - {e}");
    sys.exit(1)

def get_unified_absolute_path(waypoints_middle_only, model):
    """
    ✨【统一编码策略】与对比脚本完全一致的路径转换函数
    
    输入: waypoints_middle_only - 只包含中间航路点（不含起止点，已有序）
    输出: 包含起止点的完整绝对路径（用于绘图）
    """
    start_pos, end_pos = model['start'], model['end']
    n_wp = waypoints_middle_only.shape[0]

    # ✨【统一编码】球坐标累积变换 → 路径天然有序，无需排序！
    sorted_waypoints_relative = waypoints_middle_only

    # 🔧 MATLAB风格：先拼接起止点，再进行高度转换
    # 对应PlotSolution.m中的路径构建逻辑
    path_nodes_relative = np.vstack([start_pos, sorted_waypoints_relative, end_pos])

    # 🔧 【MATLAB对齐】使用round()索引地形矩阵，而非插值
    # PlotSolution.m line 34-36: z_map = model.H(round(y_all(i)),round(x_all(i)))
    ground_z = np.zeros(len(path_nodes_relative))
    for i in range(len(path_nodes_relative)):
        y_idx = int(np.round(path_nodes_relative[i, 1]))
        x_idx = int(np.round(path_nodes_relative[i, 0]))
        # 确保索引在有效范围内
        y_idx = np.clip(y_idx, 0, model['map_range'][1] - 1)
        x_idx = np.clip(x_idx, 0, model['map_range'][0] - 1)
        ground_z[i] = model['H'][y_idx, x_idx]

    # 高度转换：相对高度 + 地面高度 = 绝对高度
    path_nodes_absolute = path_nodes_relative.copy()
    path_nodes_absolute[:, 2] += ground_z

    # 🔧 修改说明：去掉B样条插值，直接返回12个点
    # 原因1：修改J3后，成本函数检查的是12个点的直线段
    # 原因2：B样条插值可能微调点的位置，导致成本计算和绘图不一致
    # 原因3：保证 终端成本 = 绘图路径成本
    return path_nodes_absolute


def plot_aimopso_with_custom_color(paths_absolute, path_labels, model, save_dir, scene_name, dpi=300):
    """
    使用与对比脚本相同的A*IMOPSO颜色绘制图片
    """
    # 确保使用A*IMOPSO标签，函数会自动将其重新排序到第一位并使用品红色
    path_labels = ["A*IMOPSO"]
    
    plot_and_save_paper_figures(
        paths_absolute=paths_absolute,
        path_labels=path_labels,
        model=model,
        save_dir=save_dir,
        scene_name=scene_name,
        dpi=dpi
    )




if __name__ == '__main__':
    # ==================== 【配置区域】 ====================
    SCENE_TO_RUN = 4  # 🔧 修改场景：1, 2, 3, 4
    SEED = 42          # 随机种子
    
    # 缓存设置
    USE_CACHE = True   # 🔧 是否使用缓存：True=使用缓存, False=强制重新运行
    CLEAR_CACHE = False  # 🔧 是否清除缓存：True=清除A*IMOPSO缓存后退出
    EXPERIMENT_GROUP = None  # 实验组（用于缓存区分，可设为1或2，None表示独立缓存）
    
    # 图片保存设置
    SAVE_DIR = "aimopso_diverse_solutions"  # 保存目录
    DPI = 300         # 图片分辨率
    # ====================================================
    
    print("=" * 80);
    print(" A*IMOPSO 多样化解集可视化");
    print("=" * 80)
    print(f"场景: {SCENE_TO_RUN}, 种子: {SEED}, 保存目录: {SAVE_DIR}")
    print(f"缓存: {'启用' if USE_CACHE else '禁用'}")

    # 全局参数
    # 🎯 与analyze_all_results.py和compare_algorithms_visual.py保持一致的过滤逻辑
    FEASIBILITY_THRESHOLD = 900000  # 过滤不可行解(惩罚值1e6)
    
    # 初始化缓存管理器
    cache_mgr = AlgorithmCacheManager()
    
    # 清除缓存选项
    if CLEAR_CACHE:
        print("\n🗑️  正在清除 A*IMOPSO 缓存...")
        cache_mgr.clear_cache(algorithm_name='A*IMOPSO')
        print("✅ 缓存已清除，程序退出。")
        sys.exit(0)
    
    # 算法参数（与对比脚本保持一致）
    common_params = {
        'pop_size': 100,
        'n_gen': 500,
        'seed': SEED
    }

    print("\n[1/4] 正在创建问题环境...")
    if SCENE_TO_RUN == 1:
        model = create_scene1_model()
    elif SCENE_TO_RUN == 2:
        model = create_scene2_model()
    elif SCENE_TO_RUN == 3:
        model = create_scene3_model()
    elif SCENE_TO_RUN == 4:
        model = create_scene4_model()
    else:
        raise ValueError(f"未知的场景: {SCENE_TO_RUN}。请在脚本中设置 SCENE_TO_RUN 为 1, 2, 或 3。")

    print(f"--- 已加载场景: {model.get('name', '未知')} ---")
    model['terrain_interpolator'] = RegularGridInterpolator(
        (np.arange(model['map_range'][1]), np.arange(model['map_range'][0])), model['H'], bounds_error=False,
        fill_value=0)
    var_min_for_cost = {'r': 3 * np.linalg.norm(model['start'] - model['end']) / model['n'] / 9}
    
    print("\n[2/4] 正在运行 A*IMOPSO 算法...")
    
    # 检查缓存
    algorithm_name = 'A*IMOPSO'
    scene_id = SCENE_TO_RUN
    paths_aimopso, rep_feasible = None, None
    
    if USE_CACHE:
        cached_result = cache_mgr.load_result(algorithm_name, scene_id, common_params, EXPERIMENT_GROUP)
        if cached_result:
            print("  ✅ 从缓存加载 A*IMOPSO 的结果")
            paths_aimopso = cached_result['pareto_paths']
            # 重构rep_feasible格式（从缓存的成本数据）
            pareto_costs = cached_result['pareto_costs']
            rep_feasible = [{'Cost': cost} for cost in pareto_costs]
            execution_time = cached_result['time']
            print(f"     完成, 耗时: {execution_time:.2f} 秒 (缓存)")
        else:
            print("  -> 缓存中未找到结果，正在运行算法...")
    
    # 如果没有缓存结果，运行算法
    if paths_aimopso is None or rep_feasible is None:
        start_time = time.time()
        paths_aimopso, rep_feasible, _ = run_aimopso(model, seed=SEED, mode='full')
        execution_time = time.time() - start_time
        print(f"     完成, 耗时: {execution_time:.2f} 秒")
        
        # 保存到缓存
        if USE_CACHE and rep_feasible:
            all_costs = [p['Cost'] for p in rep_feasible]
            cache_data = {
                'pareto_paths': paths_aimopso,
                'pareto_costs': all_costs,
                'time': execution_time
            }
            cache_mgr.save_result(algorithm_name, scene_id, common_params, cache_data, EXPERIMENT_GROUP)
            print(f"  ✅ 已缓存 {algorithm_name} 的结果")

    all_paths_for_plot, all_labels_for_plot, final_results = [], [], {}

    if rep_feasible:
        print("\n[3/4] 正在处理帕累托最优解集...")
        
        # 🔧 直接使用算法返回的成本，与compare_algorithms_visual.py保持一致
        # rep_feasible已经是可行解，包含了Cost信息
        all_costs = [p['Cost'] for p in rep_feasible]
        
        # 🔧 简化逻辑：rep_feasible已经由算法内部过滤，但需要额外防御性检查
        # 只需要检查是否有异常的Inf值（防御性编程）
        feasible_costs = []
        feasible_indices = []
        
        for i, cost in enumerate(all_costs):
            if (cost is not None and 
                np.all(np.isfinite(cost)) and 
                np.all(np.array(cost) < FEASIBILITY_THRESHOLD)):
                feasible_costs.append(cost)
                feasible_indices.append(i)
        
        if feasible_costs:
            # 🔧 修正：从可行解中提取Pareto前沿
            from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
            feasible_costs_array = np.array(feasible_costs)
            nd_indices = NonDominatedSorting().do(feasible_costs_array, only_non_dominated_front=True)
            
            # 从Pareto前沿中选择代表解
            pareto_costs = [feasible_costs[i] for i in nd_indices]
            # 🔧 正确的索引映射：从原始路径中选择对应的帕累托路径  
            if paths_aimopso and len(paths_aimopso) == len(all_costs):
                # 先映射到有效路径，再映射到帕累托路径
                feasible_paths = [paths_aimopso[feasible_indices[i]] for i in range(len(feasible_costs))]
                pareto_raw_paths = [feasible_paths[i] for i in nd_indices]
            else:
                print(f"     警告：路径数量({len(paths_aimopso) if paths_aimopso else 0})与解数量({len(all_costs)})不匹配！")
                # 使用第一个路径作为代表
                pareto_raw_paths = [paths_aimopso[0]] * len(pareto_costs) if paths_aimopso else []
            
            print(f"     rep_feasible解数量: {len(all_costs)}, 有效解数量: {len(feasible_costs)}, Pareto前沿数量: {len(pareto_costs)}, Pareto比例: {len(pareto_costs)/len(feasible_costs):.3f}")
            
            pareto_costs_array = np.array(pareto_costs)
            total_costs = np.sum(pareto_costs_array, axis=1)

            # 1. 基于Pareto前沿筛选优秀的、多样化的解，用于终端显示
            excellent_indices = {
                "A*IMOPSO (Best Total)": np.argmin(total_costs),
                "A*IMOPSO (Shortest)": np.argmin(pareto_costs_array[:, 0]),
                "A*IMOPSO (Safest)": np.argmin(pareto_costs_array[:, 1]),
                "A*IMOPSO (Lowest)": np.argmin(pareto_costs_array[:, 2]),
                "A*IMOPSO (Smoothest)": np.argmin(pareto_costs_array[:, 3]),
            }
            for name, idx in excellent_indices.items():
                final_results[name] = pareto_costs[idx]

            # 2. 从Pareto前沿中选择总成本最低的解，用于绘图
            best_total_idx = np.argmin(total_costs)
            best_raw_path = pareto_raw_paths[best_total_idx]
            path_to_plot = get_unified_absolute_path(best_raw_path[1:-1], model)
            all_paths_for_plot.append(path_to_plot)
            all_labels_for_plot.append("A*IMOPSO")  # 移除 "(Best Total Cost)" 后缀

            # 3. 打印终端统计信息
            print("     完成。已筛选出优秀的代表性解:")
            print("-" * 80)
            print(f"{'Solution Type':<28} | {'J1 (Length)':<12} | {'J2 (Threat)':<12} | {'J3 (Height)':<12} | {'J4 (Smooth)':<12}")
            print("-" * 80)
            for name, costs in sorted(final_results.items()):
                print(f"{name:<28} | {costs[0]:<12.4f} | {costs[1]:<12.4f} | {costs[2]:<12.4f} | {costs[3]:<12.4f}")
            print("-" * 80)

        else:
            print("     警告：没有找到有效的可行解！")
    else:
        print("     警告: A*IMOPSO 未返回rep_feasible。")

    print("\n[4/4] 正在生成高质量论文图片...")
    if all_paths_for_plot:
        # 使用自定义颜色绘图函数，确保A*IMOPSO颜色与对比脚本一致
        scene_name = f"scene_{SCENE_TO_RUN}"
        
        plot_aimopso_with_custom_color(
            paths_absolute=all_paths_for_plot,
            path_labels=all_labels_for_plot, 
            model=model,
            save_dir=SAVE_DIR,
            scene_name=scene_name,
            dpi=DPI
        )
        print("     绘图完成！")
        print(f"\n✅ 图片已保存到目录: {SAVE_DIR}/")
        print(f"   - {scene_name}_3d_view.png (3D视图)")
        print(f"   - {scene_name}_top_view.png (俯视图)")
        print(f"   - {scene_name}_combined.png (PNG拼接图)")
        print(f"   - {scene_name}_combined.eps (EPS拼接图) ⭐")
    else:
        print("     警告: 未找到任何可行的路径用于绘图。")

    print("\n💾 缓存统计:")
    cache_mgr.list_cache()
    
    print("\n运行结束。")
