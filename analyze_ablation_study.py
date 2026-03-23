import numpy as np
import pandas as pd
import os
import sys
from scipy.stats import friedmanchisquare, rankdata
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as mticker

# --- Matplotlib 配置 ---
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# --- Pymoo Imports ---
try:
    from pymoo.indicators.hv import HV
    from pymoo.indicators.igd import IGD
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
except ImportError as e:
    print(f"[ERROR] Failed to import Pymoo: {e}", file=sys.stderr)
    sys.exit(1)

# --- 分析配置 ---
BASE_DIR = 'run_results_ablation'
N_RUNS = 30
SCENES = [1, 2, 3, 4]
ALGORITHMS = ['AIMOPSO', 'A_MOPSO', 'IMOPSO', 'MOPSO']
DISPLAY_NAMES = {
    'AIMOPSO': 'A*IMOPSO',
    'A_MOPSO': 'A*MOPSO',
    'IMOPSO': 'IMOPSO',
    'MOPSO': 'MOPSO'
}

def load_ablation_data(scene_num):
    """为消融研究加载单个场景的数据。"""
    data = {}
    for alg in ALGORITHMS:
        alg_dir = os.path.join(BASE_DIR, f"scene{scene_num}", alg)
        run_results = []
        if not os.path.exists(alg_dir):
            print(f"Warning: Directory not found for {alg} in scene {scene_num}")
            data[alg] = [None] * N_RUNS
            continue

        for i in range(1, N_RUNS + 1):
            file_path = os.path.join(alg_dir, f"run_{i}.csv")
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                df = pd.read_csv(file_path)
                run_results.append(df.values)
            else:
                run_results.append(None)
        data[alg] = run_results
    return data

def compute_ablation_pf_true(data):
    """根据消融研究的四种算法变体计算统一的Pareto参考前沿。"""
    all_sols = []
    for alg in ALGORITHMS:
        for run_data in data[alg]:
            if run_data is not None:
                all_sols.append(run_data)
    
    if not all_sols:
        return None

    F_all = np.vstack(all_sols)
    nd_indices = NonDominatedSorting().do(F_all, only_non_dominated_front=True)
    return F_all[nd_indices]

def calculate_metrics(data, pf_true):
    """计算IGD和HV指标。"""
    igd_calculator = IGD(pf=pf_true)
    # HV的参考点需要根据真实PF动态调整
    ref_point = np.max(pf_true, axis=0) * 1.1
    hv_calculator = HV(ref_point=ref_point)

    metrics = {alg: {'igd': [], 'hv': []} for alg in ALGORITHMS}
    for alg in ALGORITHMS:
        for run_data in data[alg]:
            if run_data is not None:
                metrics[alg]['igd'].append(igd_calculator.do(run_data))
                metrics[alg]['hv'].append(hv_calculator.do(run_data))
            else:
                metrics[alg]['igd'].append(np.nan)
                metrics[alg]['hv'].append(np.nan)
    return metrics

def compute_scene_ranks(metrics, metric_name, higher_is_better=True):
    """Compute average ranks for a single scene using rankdata for consistency."""
    algorithms = list(metrics.keys())
    
    # Find the minimum number of runs across all algorithms for this scene
    min_runs = min(len(metrics[alg][metric_name]) for alg in algorithms if metrics[alg][metric_name])
    if min_runs == 0:
        return pd.Series(index=algorithms, dtype=float)

    # Assemble data for ranking (runs x algorithms)
    data_matrix = np.array([metrics[alg][metric_name][:min_runs] for alg in algorithms]).T

    # Rank each run (row)
    all_run_ranks = []
    for run_data in data_matrix:
        # For HV, higher is better, so we rank the negative of the data
        if higher_is_better:
            ranks = rankdata(-run_data, method='average')
        else:
            ranks = rankdata(run_data, method='average')
        all_run_ranks.append(ranks)
    
    # Calculate average rank for each algorithm
    avg_ranks = np.mean(all_run_ranks, axis=0)
    return pd.Series(avg_ranks, index=algorithms)

def plot_friedman_ranks(all_scenes_metrics, save_path='ablation_friedman_ranks.png'):
    """绘制与SOTA对比图风格完全一致的Friedman排名图。"""
    # Calculate and average ranks
    scene_igd_ranks = {s: compute_scene_ranks(m, 'igd', False) for s, m in all_scenes_metrics.items()}
    scene_hv_ranks = {s: compute_scene_ranks(m, 'hv', True) for s, m in all_scenes_metrics.items()}
    avg_igd_ranks = pd.DataFrame(scene_igd_ranks).mean(axis=1).reindex(ALGORITHMS)
    avg_hv_ranks = pd.DataFrame(scene_hv_ranks).mean(axis=1).reindex(ALGORITHMS)

    n_algs = len(ALGORITHMS)
    fig, ax = plt.subplots(figsize=(11, 5.5))

    # --- 精确颜色匹配 ---
    colors = {
        'AIMOPSO': '#EB987A',  # 浅橙/珊瑚
        'A_MOPSO': '#F8E4DE',  # 淡粉白
        'IMOPSO': '#DC6C6E',   # 玫瑰红
        'MOPSO': '#F2B5B8'    # 浅粉
    }

    # --- 布局设置 ---
    bar_width = 0.8
    group_gap = 2

    # --- 绘制IGD排名 ---
    igd_positions = np.arange(n_algs)
    for i, alg in enumerate(ALGORITHMS):
        bar = ax.bar(igd_positions[i], avg_igd_ranks[alg], bar_width, 
                     color=colors[alg], label=DISPLAY_NAMES[alg], edgecolor='black', linewidth=0.7)
        ax.bar_label(bar, fmt='%.2f', padding=3, fontsize=14)

    # --- 绘制HV排名 ---
    hv_positions = igd_positions + n_algs + group_gap
    for i, alg in enumerate(ALGORITHMS):
        bar = ax.bar(hv_positions[i], avg_hv_ranks[alg], bar_width, 
                     color=colors[alg], edgecolor='black', linewidth=0.7)
        ax.bar_label(bar, fmt='%.2f', padding=3, fontsize=14)

    # --- 添加中心分隔线和指标标签 ---
    separator_pos = n_algs + group_gap / 2 - 0.5
    ax.axvline(separator_pos, color='grey', linestyle='--', linewidth=1.2)
    max_rank_val = max(avg_igd_ranks.max(), avg_hv_ranks.max()) if not avg_igd_ranks.empty else 1
    text_y_pos = -0.02 * ax.get_ylim()[1]
    tick_len = 0.01 * ax.get_ylim()[1]

    # IGD Label and Tick
    igd_center = (igd_positions[0] + igd_positions[-1]) / 2
    ax.text(igd_center, text_y_pos, 'IGD', ha='center', va='top', fontsize=20)
    ax.plot([igd_center, igd_center], [0, -tick_len], color='black', lw=1.2)

    # HV Label and Tick
    hv_center = (hv_positions[0] + hv_positions[-1]) / 2
    ax.text(hv_center, text_y_pos, 'HV', ha='center', va='top', fontsize=20)
    ax.plot([hv_center, hv_center], [0, -tick_len], color='black', lw=1.2)

    # --- 精确美化图表 ---
    ax.set_ylabel('Average Rank', fontsize=18)
    ax.set_title('') # No title
    ax.set_xticks([])
    ax.tick_params(axis='y', labelsize=16)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1)) # Y-axis ticks every 1 unit
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.5))
    ax.grid(axis='y', linestyle=':', alpha=0.7, which='major')
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['top'].set_linewidth(1.2)
    ax.spines['right'].set_linewidth(1.2)
    ax.set_ylim(0, 5)

    # --- 图例 ---
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', 
              ncol=1, fancybox=True, shadow=False, fontsize=14)

    plt.tight_layout()
    
    # 保存 PNG 格式
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 保存 EPS 格式
    save_path_eps = save_path.replace('.png', '.eps')
    plt.savefig(save_path_eps, format='eps', bbox_inches='tight')
    
    print(f"✓ Friedman排名图已保存:")
    print(f"  - PNG: {save_path}")
    print(f"  - EPS: {save_path_eps}")
    
    plt.close()

def main():
    all_scenes_metrics = {}
    for scene in SCENES:
        print(f"--- 正在分析场景 {scene} ---")
        data = load_ablation_data(scene)
        pf_true = compute_ablation_pf_true(data)
        if pf_true is None:
            print(f"场景 {scene} 无有效数据，跳过分析。")
            continue
        print(f"场景 {scene} 的参考前沿包含 {pf_true.shape[0]} 个解。")
        
        metrics = calculate_metrics(data, pf_true)
        all_scenes_metrics[scene] = metrics

        # 打印当前场景的平均指标
        print("平均指标值:")
        for alg in ALGORITHMS:
            avg_igd = np.nanmean(metrics[alg]['igd'])
            avg_hv = np.nanmean(metrics[alg]['hv'])
            print(f"  {DISPLAY_NAMES[alg]:<10}: IGD={avg_igd:.4f}, HV={avg_hv:.4f}")

    # 跨场景Friedman检验与绘图
    if all_scenes_metrics:
        plot_friedman_ranks(all_scenes_metrics)
    else:
        print("未能收集到任何有效数据进行最终分析。")

if __name__ == '__main__':
    main()
