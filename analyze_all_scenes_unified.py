"""
多场景统一分析脚本（完整版）
- 一次性分析所有场景（1-4）
- 计算 J1-J4 基础统计
- 生成跨场景的 Friedman 排名图
- 统计跨场景的 +/≈/- 总计
- 输出论文格式的表格
"""

import numpy as np
import pandas as pd
import os
import sys
from scipy.stats import ranksums, friedmanchisquare
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# --- Pymoo Imports ---
try:
    from pymoo.indicators.hv import Hypervolume
    from pymoo.indicators.igd import IGD
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
except ImportError as e:
    print(f"[ERROR] Failed to import Pymoo: {e}", file=sys.stderr)
    sys.exit(1)


def load_scene_data(scene_num, algorithms, n_runs, results_dirs):
    """Load data for a single scene."""
    algorithm_data = {}
    
    for alg in algorithms:
        alg_dir = None
        for results_dir in results_dirs:
            potential_dir = os.path.join(results_dir, f"scene_{scene_num}", alg)
            if os.path.exists(potential_dir):
                alg_dir = potential_dir
                break
        
        if alg_dir is None:
            algorithm_data[alg] = []
            continue
        
        run_results = []
        for i in range(1, n_runs + 1):
            file_path = os.path.join(alg_dir, f"run_{i}.csv")
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                try:
                    df = pd.read_csv(file_path)
                    if not df.empty:
                        feasible_solutions = df[(df < 900000).all(axis=1)].values
                        if feasible_solutions.shape[0] > 0:
                            run_results.append(feasible_solutions)
                        else:
                            run_results.append(None)
                    else:
                        run_results.append(None)
                except Exception:
                    run_results.append(None)
            else:
                run_results.append(None)
        
        algorithm_data[alg] = run_results
    
    return algorithm_data


def compute_unified_reference_front(all_algorithm_data):
    """Compute unified reference Pareto front from ALL algorithms."""
    all_successful_sols = []
    for results_list in all_algorithm_data.values():
        for res in results_list:
            if res is not None:
                all_successful_sols.append(res)

    if not all_successful_sols:
        return None, None, None, None

    F_all = np.vstack(all_successful_sols)
    nd_indices = NonDominatedSorting().do(F_all, only_non_dominated_front=True)
    PF_approx = F_all[nd_indices]

    min_vals = np.min(PF_approx, axis=0)
    max_vals = np.max(PF_approx, axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals < 1e-9] = 1.0
    PF_approx_norm = (PF_approx - min_vals) / range_vals
    
    return PF_approx_norm, min_vals, range_vals, PF_approx


def compute_metrics_with_unified_front(algorithm_data, PF_approx_norm, min_vals, range_vals):
    """Compute HV and IGD metrics using the unified reference front."""
    ref_point_hv = np.array([1.1] * 4)
    hv_calculator = Hypervolume(ref_point=ref_point_hv, normalize=False)
    igd_calculator = IGD(pf=PF_approx_norm, zero_to_one=True)

    metrics = {}
    for alg_name, results_list in algorithm_data.items():
        successful_runs = [r for r in results_list if r is not None]
        if not successful_runs:
            metrics[alg_name] = {'hv': [], 'igd': []}
            continue

        hv_scores, igd_scores = [], []
        for front in successful_runs:
            nd_indices = NonDominatedSorting().do(front, only_non_dominated_front=True)
            pareto_front = front[nd_indices]
            
            if pareto_front.shape[0] > 0:
                front_norm = (pareto_front - min_vals) / range_vals
                hv_scores.append(hv_calculator.do(front_norm))
                igd_scores.append(igd_calculator.do(front_norm))
        
        metrics[alg_name] = {'hv': hv_scores, 'igd': igd_scores}
    
    return metrics


def compute_objective_statistics(algorithm_data, n_runs):
    """
    Compute J1-J4 statistics for each algorithm.
    These metrics are computed independently for each algorithm (no cross-algorithm comparison).
    """
    stats_summary = {}
    
    for alg_name, results_list in algorithm_data.items():
        successful_runs = [r for r in results_list if r is not None]
        success_rate = len(successful_runs) / n_runs
        
        if not successful_runs:
            stats_summary[alg_name] = {
                'success_rate': 0.0,
                'objectives': {f'J{j+1}': {'min': 0, 'max': 0, 'mean': 0, 'std': 0} for j in range(4)}
            }
            continue
        
        # Compute J1-J4 statistics based on Pareto fronts from each run
        pareto_stats = {f'J{j+1}': {'min': [], 'max': [], 'mean': [], 'std': []} for j in range(4)}
        
        for run_data in successful_runs:
            # Extract Pareto front for this run
            nd_indices = NonDominatedSorting().do(run_data, only_non_dominated_front=True)
            pareto_front = run_data[nd_indices]
            
            if pareto_front.shape[0] > 0:
                # Compute statistics for each objective
                for j in range(4):
                    obj_values = pareto_front[:, j]
                    pareto_stats[f'J{j+1}']['min'].append(np.min(obj_values))
                    pareto_stats[f'J{j+1}']['max'].append(np.max(obj_values))
                    pareto_stats[f'J{j+1}']['mean'].append(np.mean(obj_values))
                    pareto_stats[f'J{j+1}']['std'].append(np.std(obj_values))
        
        # Average statistics across all runs
        objectives = {}
        for j in range(4):
            obj_key = f'J{j+1}'
            if pareto_stats[obj_key]['min']:
                objectives[obj_key] = {
                    'min': np.mean(pareto_stats[obj_key]['min']),
                    'max': np.mean(pareto_stats[obj_key]['max']),
                    'mean': np.mean(pareto_stats[obj_key]['mean']),
                    'std': np.mean(pareto_stats[obj_key]['std'])
                }
            else:
                objectives[obj_key] = {'min': 0, 'max': 0, 'mean': 0, 'std': 0}
        
        stats_summary[alg_name] = {
            'success_rate': success_rate,
            'objectives': objectives
        }
    
    return stats_summary


def print_objective_statistics_table(all_scenes_obj_stats, algorithms, display_names, scenes):
    """Print J1-J4 statistics table for all scenes."""
    print("\n" + "=" * 150)
    print(" 基础目标统计 (J1-J4) - 所有场景")
    print("=" * 150)
    
    for scene_num in scenes:
        print(f"\n场景 {scene_num}:")
        print("-" * 150)
        
        headers = ['Algorithm', 'Success', 
                   'J1_Min', 'J1_Max', 'J1_Mean', 'J1_Std',
                   'J2_Min', 'J2_Max', 'J2_Mean', 'J2_Std',
                   'J3_Min', 'J3_Max', 'J3_Mean', 'J3_Std',
                   'J4_Min', 'J4_Max', 'J4_Mean', 'J4_Std']
        
        print(' | '.join([f'{h:<10}' for h in headers]))
        print('-' * 200)
        
        stats_summary = all_scenes_obj_stats[scene_num]
        for alg_name in algorithms:
            if alg_name not in stats_summary:
                continue
                
            stats = stats_summary[alg_name]
            display_name = display_names.get(alg_name, alg_name)
            row = [f'{display_name:<10}', f'{stats["success_rate"]:<10.2f}']
            
            for j in range(4):
                obj_key = f'J{j+1}'
                obj_stats = stats['objectives'][obj_key]
                row.extend([
                    f'{obj_stats["min"]:<10.4f}',
                    f'{obj_stats["max"]:<10.4f}',
                    f'{obj_stats["mean"]:<10.4f}',
                    f'{obj_stats["std"]:<10.4f}'
                ])
            
            print(' | '.join(row))
    
    print("\n" + "=" * 150)
    print("指标说明:")
    print("  J1 (路径成本), J2 (威胁成本), J3 (高度成本), J4 (平滑度成本) - 越小越好")
    print("  注: 这些指标基于每个算法自身的 30 次运行，不涉及算法间比较")
    print("=" * 150)


def compute_scene_ranks(metrics, metric_name, higher_is_better=True):
    """Compute average ranks for a single scene."""
    algorithms = list(metrics.keys())
    n_algs = len(algorithms)
    
    # Get minimum length
    min_length = min(len(metrics[alg][metric_name]) for alg in algorithms if len(metrics[alg][metric_name]) > 0)
    if min_length == 0:
        return None
    
    # Compute ranks for each run
    n_runs = min_length
    ranks = np.zeros((n_runs, n_algs))
    
    for i in range(n_runs):
        values = [metrics[alg][metric_name][i] for alg in algorithms]
        if higher_is_better:
            sorted_indices = np.argsort(values)[::-1]  # Descending
        else:
            sorted_indices = np.argsort(values)  # Ascending
        
        for rank, idx in enumerate(sorted_indices, 1):
            ranks[i, idx] = rank
    
    avg_ranks = np.mean(ranks, axis=0)
    return avg_ranks


def wilcoxon_test(data1, data2):
    """Perform Wilcoxon rank-sum test."""
    if len(data1) == 0 or len(data2) == 0:
        return 1.0
    try:
        stat, p_value = ranksums(data1, data2)
        return p_value
    except Exception:
        return 1.0


def compare_single_scene(metrics, reference_alg, metric_name, higher_is_better=True, alpha=0.05):
    """Compare reference algorithm against all others for a single scene."""
    ref_data = np.array(metrics[reference_alg][metric_name])
    results = {}
    
    for alg_name in metrics.keys():
        if alg_name == reference_alg:
            results[alg_name] = {
                'symbol': '', 
                'p_value': 1.0, 
                'mean': np.mean(ref_data), 
                'std': np.std(ref_data)
            }
            continue
        
        comp_data = np.array(metrics[alg_name][metric_name])
        
        if len(ref_data) == 0 or len(comp_data) == 0:
            results[alg_name] = {'symbol': '≈', 'p_value': 1.0, 'mean': 0, 'std': 0}
            continue
        
        p_value = wilcoxon_test(ref_data, comp_data)
        mean_ref = np.mean(ref_data)
        mean_comp = np.mean(comp_data)
        
        if p_value >= alpha:
            symbol = '≈'
        else:
            if higher_is_better:
                symbol = '+' if mean_ref > mean_comp else '-'
            else:
                symbol = '+' if mean_ref < mean_comp else '-'
        
        results[alg_name] = {
            'symbol': symbol, 
            'p_value': p_value,
            'mean': mean_comp,
            'std': np.std(comp_data)
        }
    
    return results


def plot_cross_scene_friedman_ranks(all_scenes_ranks, algorithms, display_names, save_path_base=None):
    """
    Plot Friedman average ranks across all scenes.
    Generates both Chinese and English versions.
    X-axis: IGD and HV (two groups)
    Y-axis: Average rank across all scenes
    """
    # Compute average ranks across scenes
    hv_avg_ranks = np.mean([ranks['hv'] for ranks in all_scenes_ranks.values()], axis=0)
    igd_avg_ranks = np.mean([ranks['igd'] for ranks in all_scenes_ranks.values()], axis=0)
    
    # Get algorithm names and colors
    alg_names = [display_names.get(alg, alg) for alg in algorithms]
    n_algs = len(alg_names)
    
    colors = ['#EB987A', '#F8E4DE', '#DC6C6E', '#BBD5D4', '#EFE9D3', '#D7EAEC', '#C8B8D4']
    
    bar_width = 0.12
    
    # IGD group (left side)
    igd_positions = np.arange(n_algs) * bar_width
    igd_center = np.mean(igd_positions)
    
    # HV group (right side)
    hv_positions = igd_positions + (n_algs + 2) * bar_width
    hv_center = np.mean(hv_positions)
    
    max_rank = max(max(igd_avg_ranks), max(hv_avg_ranks))
    separator_x = (igd_positions[-1] + hv_positions[0]) / 2
    
    # Generate both versions
    for lang in ['en', 'zh']:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot IGD bars
        for i, (rank, name) in enumerate(zip(igd_avg_ranks, alg_names)):
            ax.bar(igd_positions[i], rank, bar_width, 
                   color=colors[i % len(colors)], alpha=0.85,
                   label=name if i < n_algs else "")
            ax.text(igd_positions[i], rank + 0.1, f'{rank:.2f}', 
                   ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Plot HV bars
        for i, (rank, name) in enumerate(zip(hv_avg_ranks, alg_names)):
            ax.bar(hv_positions[i], rank, bar_width, 
                  color=colors[i % len(colors)], alpha=0.85)
            ax.text(hv_positions[i], rank + 0.1, f'{rank:.2f}', 
                   ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Set labels based on language
        if lang == 'zh':
            ax.set_ylabel('平均排名', fontsize=18, fontweight='bold')
        else:  # English
            ax.set_ylabel('Average Rank', fontsize=18, fontweight='bold')
        
        # No title - removed as requested
        
        ax.set_xticks([igd_center, hv_center])
        ax.set_xticklabels(['IGD', 'HV'], fontsize=20, fontweight='bold')
        
        ax.legend(loc='upper right', fontsize=14, ncol=2, framealpha=0.9)
        ax.tick_params(axis='y', labelsize=16)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
        
        ax.set_ylim(0, max_rank * 1.2)
        
        ax.axvline(x=separator_x, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        
        plt.tight_layout()
        
        if save_path_base:
            # Generate filename with language suffix
            if lang == 'zh':
                save_path_png = save_path_base.replace('.png', '_zh.png')
                save_path_eps = save_path_base.replace('.png', '_zh.eps')
                lang_name = '中文版'
            else:
                save_path_png = save_path_base.replace('.png', '_en.png')
                save_path_eps = save_path_base.replace('.png', '_en.eps')
                lang_name = 'English'
            
            # 保存 PNG 格式
            plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
            print(f"✓ PNG 图表已保存 ({lang_name}): {save_path_png}")
            
            # 保存 EPS 格式
            plt.savefig(save_path_eps, format='eps', bbox_inches='tight')
            print(f"✓ EPS 图表已保存 ({lang_name}): {save_path_eps}")
        
        plt.close(fig)
    
    return None


def print_cross_scene_comparison_table(all_scenes_comparisons, algorithms, reference_alg, 
                                       display_names, group_name, scenes):
    """Print comparison table across all scenes."""
    print("\n" + "=" * 120)
    print(f" {group_name}")
    print("=" * 120)
    
    # Header
    header = f"{'Scene':<8} {'Metric':<8} "
    for alg in algorithms:
        header += f"{display_names.get(alg, alg):<20} "
    print(header)
    print("-" * 120)
    
    # Data rows
    for scene_num in scenes:
        hv_comp = all_scenes_comparisons[scene_num]['hv']
        igd_comp = all_scenes_comparisons[scene_num]['igd']
        
        # HV row
        row_hv = f"{scene_num:<8} {'HV':<8} "
        for alg in algorithms:
            result = hv_comp[alg]
            value_str = f"{result['mean']:.4e}({result['std']:.2e}){result['symbol']}"
            row_hv += f"{value_str:<20} "
        print(row_hv)
        
        # IGD row
        row_igd = f"{'':8} {'IGD':<8} "
        for alg in algorithms:
            result = igd_comp[alg]
            value_str = f"{result['mean']:.4e}({result['std']:.2e}){result['symbol']}"
            row_igd += f"{value_str:<20} "
        print(row_igd)
        print()
    
    # Calculate cross-scene +/≈/- statistics
    print("-" * 120)
    
    hv_stats = {}
    igd_stats = {}
    
    for alg in algorithms:
        if alg == reference_alg:
            hv_stats[alg] = {'plus': 0, 'approx': 0, 'minus': 0}
            igd_stats[alg] = {'plus': 0, 'approx': 0, 'minus': 0}
            continue
        
        hv_plus = sum(1 for s in scenes if all_scenes_comparisons[s]['hv'][alg]['symbol'] == '+')
        hv_approx = sum(1 for s in scenes if all_scenes_comparisons[s]['hv'][alg]['symbol'] == '≈')
        hv_minus = sum(1 for s in scenes if all_scenes_comparisons[s]['hv'][alg]['symbol'] == '-')
        
        igd_plus = sum(1 for s in scenes if all_scenes_comparisons[s]['igd'][alg]['symbol'] == '+')
        igd_approx = sum(1 for s in scenes if all_scenes_comparisons[s]['igd'][alg]['symbol'] == '≈')
        igd_minus = sum(1 for s in scenes if all_scenes_comparisons[s]['igd'][alg]['symbol'] == '-')
        
        hv_stats[alg] = {'plus': hv_plus, 'approx': hv_approx, 'minus': hv_minus}
        igd_stats[alg] = {'plus': igd_plus, 'approx': igd_approx, 'minus': igd_minus}
    
    # Print +/≈/- row
    row_hv_stat = f"{'+ / ≈ / -':<8} {'HV':<8} "
    for alg in algorithms:
        if alg == reference_alg:
            row_hv_stat += f"{'-':<20} "
        else:
            stat_str = f"{hv_stats[alg]['plus']}/{hv_stats[alg]['approx']}/{hv_stats[alg]['minus']}"
            row_hv_stat += f"{stat_str:<20} "
    print(row_hv_stat)
    
    row_igd_stat = f"{'':8} {'IGD':<8} "
    for alg in algorithms:
        if alg == reference_alg:
            row_igd_stat += f"{'-':<20} "
        else:
            stat_str = f"{igd_stats[alg]['plus']}/{igd_stats[alg]['approx']}/{igd_stats[alg]['minus']}"
            row_igd_stat += f"{stat_str:<20} "
    print(row_igd_stat)
    
    print("\n符号说明:")
    print(f"  '+': {display_names[reference_alg]} 显著优于该算法 (Wilcoxon 检验, p < 0.05)")
    print(f"  '≈': 两者无显著差异 (p ≥ 0.05)")
    print(f"  '-': {display_names[reference_alg]} 显著劣于该算法")
    print(f"  '+/≈/-': 在 {len(scenes)} 个场景中的统计")


if __name__ == "__main__":
    # --- Configuration ---
    SCENES_TO_RUN = [1, 2, 3, 4]
    N_RUNS = 30
    REFERENCE_ALGORITHM = "AIMOPSO"
    ALPHA = 0.05
    
    DISPLAY_NAMES = {
        "AIMOPSO": "A*IMOPSO",
        "NSGA-II": "NSGA-II", 
        "SPEA2": "SPEA2",
        "SMS-EMOA": "SMS-EMOA",
        "NSGA-III": "NSGA-III",
        "RVEA": "RVEA",
        "AGEMOEA2": "AGEMOEA2"
    }
    
    ALL_ALGORITHMS = ["AIMOPSO", "NSGA-II", "SPEA2", "SMS-EMOA", "NSGA-III", "RVEA", "AGEMOEA2"]
    CLASSIC_ALGORITHMS = ["AIMOPSO", "NSGA-II", "SPEA2", "SMS-EMOA"]
    SOTA_ALGORITHMS = ["AIMOPSO", "NSGA-III", "RVEA", "AGEMOEA2"]
    
    RESULTS_DIRS = ["run_results_mo", "run_results_mo_sota"]
    # ----------------------------

    print(f"\n{'='*120}")
    print(f" 多场景统一分析 (场景 {SCENES_TO_RUN})")
    print(f" 参考算法: {DISPLAY_NAMES[REFERENCE_ALGORITHM]}")
    print(f"{'='*120}\n")

    # Step 1: Load and process all scenes
    print("█" * 120)
    print(" 步骤 1: 加载所有场景数据并计算指标")
    print("█" * 120)
    
    all_scenes_metrics = {}
    all_scenes_ranks = {}
    all_scenes_comparisons = {}
    all_scenes_obj_stats = {}
    
    for scene_num in SCENES_TO_RUN:
        print(f"\n处理场景 {scene_num}...")
        
        # Load data
        data = load_scene_data(scene_num, ALL_ALGORITHMS, N_RUNS, RESULTS_DIRS)
        
        # Compute unified reference front
        PF_approx_norm, min_vals, range_vals, PF_approx = compute_unified_reference_front(data)
        
        if PF_approx_norm is None:
            print(f"  场景 {scene_num}: 无法计算参考前沿，跳过")
            continue
        
        print(f"  场景 {scene_num}: 参考前沿包含 {PF_approx.shape[0]} 个解")
        
        # Compute objective statistics (J1-J4, SD)
        obj_stats = compute_objective_statistics(data, N_RUNS)
        all_scenes_obj_stats[scene_num] = obj_stats
        
        # Compute metrics (HV, IGD)
        metrics = compute_metrics_with_unified_front(data, PF_approx_norm, min_vals, range_vals)
        all_scenes_metrics[scene_num] = metrics
        
        # Compute ranks
        hv_ranks = compute_scene_ranks(metrics, 'hv', higher_is_better=True)
        igd_ranks = compute_scene_ranks(metrics, 'igd', higher_is_better=False)
        all_scenes_ranks[scene_num] = {'hv': hv_ranks, 'igd': igd_ranks}
        
        # Compute comparisons
        hv_comp = compare_single_scene(metrics, REFERENCE_ALGORITHM, 'hv', higher_is_better=True, alpha=ALPHA)
        igd_comp = compare_single_scene(metrics, REFERENCE_ALGORITHM, 'igd', higher_is_better=False, alpha=ALPHA)
        all_scenes_comparisons[scene_num] = {'hv': hv_comp, 'igd': igd_comp}
        
        print(f"  场景 {scene_num}: ✓ 完成")
    
    # Step 2: Print objective statistics (J1-J4, SD)
    print("\n" + "█" * 120)
    print(" 步骤 2: 基础目标统计 (J1-J4) 和多样性指标 (SD)")
    print("█" * 120)
    
    print_objective_statistics_table(all_scenes_obj_stats, ALL_ALGORITHMS, DISPLAY_NAMES, SCENES_TO_RUN)
    
    # Step 3: Friedman test and ranking plot across all scenes
    print("\n" + "█" * 120)
    print(" 步骤 3: Friedman 检验（基于所有场景）")
    print("█" * 120)
    
    plot_path_base = f"friedman_ranks_all_scenes.png"
    plot_cross_scene_friedman_ranks(all_scenes_ranks, ALL_ALGORITHMS, DISPLAY_NAMES, plot_path_base)
    
    # Compute overall Friedman test
    hv_avg_ranks = np.mean([ranks['hv'] for ranks in all_scenes_ranks.values()], axis=0)
    igd_avg_ranks = np.mean([ranks['igd'] for ranks in all_scenes_ranks.values()], axis=0)
    
    print(f"\n跨场景平均排名:")
    for i, alg in enumerate(ALL_ALGORITHMS):
        print(f"  {DISPLAY_NAMES[alg]:<15} - IGD: {igd_avg_ranks[i]:.2f}, HV: {hv_avg_ranks[i]:.2f}")
    
    # Step 4: Print comparison tables
    print("\n" + "█" * 120)
    print(" 步骤 4: Wilcoxon 两两比较（分组统计）")
    print("█" * 120)
    
    print_cross_scene_comparison_table(all_scenes_comparisons, CLASSIC_ALGORITHMS, 
                                      REFERENCE_ALGORITHM, DISPLAY_NAMES, 
                                      "6.3.1 与经典多目标算法的对比", SCENES_TO_RUN)
    
    print_cross_scene_comparison_table(all_scenes_comparisons, SOTA_ALGORITHMS, 
                                      REFERENCE_ALGORITHM, DISPLAY_NAMES, 
                                      "6.3.2 与最新SOTA算法的对比", SCENES_TO_RUN)
    
    print("\n" + "="*120)
    print(" ✓ 多场景分析完成！")
    print("="*120)
    print(f"\n输出文件:")
    print(f"  - friedman_ranks_all_scenes_zh.png (Friedman 排名图 - 中文版)")
    print(f"  - friedman_ranks_all_scenes_zh.eps (Friedman 排名图 - 中文版 EPS)")
    print(f"  - friedman_ranks_all_scenes_en.png (Friedman 排名图 - English)")
    print(f"  - friedman_ranks_all_scenes_en.eps (Friedman 排名图 - English EPS)")
