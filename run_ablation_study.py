import os
import numpy as np
import pandas as pd
import argparse
from environments import create_scene1_model, create_scene2_model, create_scene3_model, create_scene4_model
from aimopso_runner import run_aimopso

# --- 配置 ---
BASE_DIR = 'run_results_ablation'
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

SCENES = [1, 2, 3, 4]
ALGORITHMS = {
    'A_IMOPSO': {'use_a_star_init': True, 'use_dual_leader': True},
    'A_MOPSO': {'use_a_star_init': True, 'use_dual_leader': False},
    'IMOPSO': {'use_a_star_init': False, 'use_dual_leader': True},
    'MOPSO': {'use_a_star_init': False, 'use_dual_leader': False}
}

def run_single_experiment(args):
    algo_name, scene, run_id, params, end_run = args
    scene_name = f'scene{scene}'
    output_dir = os.path.join(BASE_DIR, scene_name, algo_name)
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, f'run_{run_id}.csv')

    if os.path.exists(result_file):
        print(f"Skipping {algo_name} on {scene_name}, run {run_id} (already exists)")
        return None

    print(f"Running {algo_name} on {scene_name}, run {run_id}/{end_run}")

    scene_creators = {
        1: create_scene1_model,
        2: create_scene2_model,
        3: create_scene3_model,
        4: create_scene4_model
    }
    model = scene_creators[scene]()
    seed = run_id
    
    costs = run_aimopso(model, seed=seed, mode='stats', 
                        use_a_star_init=params['use_a_star_init'], 
                        use_dual_leader=params['use_dual_leader'])
    
    # 保存原始成本向量
    if costs is not None and costs.shape[0] > 0:
        df = pd.DataFrame(costs, columns=[f'J{j+1}' for j in range(costs.shape[1])])
        df.to_csv(result_file, index=False)
    else:
        # 创建一个空文件表示失败的运行
        open(result_file, 'w').close()

    return f"Finished {algo_name} on {scene_name}, run {run_id}"

def main():
    parser = argparse.ArgumentParser(description='Run ablation study for MOPSO variants.')
    parser.add_argument('--algorithm', type=str, choices=ALGORITHMS.keys(), required=True,
                        help='Specify a single algorithm to run.')
    parser.add_argument('--scene', type=int, choices=SCENES, 
                        help='Specify a single scene to run. If not provided, all scenes will be run.')
    parser.add_argument('--start_run', type=int, default=1, help='The starting run number (inclusive).')
    parser.add_argument('--end_run', type=int, default=30, help='The ending run number (inclusive).')
    args = parser.parse_args()

    algorithms_to_run = [args.algorithm]
    scenes_to_run = [args.scene] if args.scene else SCENES

    tasks = []
    for algo_name in algorithms_to_run:
        params = ALGORITHMS[algo_name]
        for scene in scenes_to_run:
            for i in range(args.start_run, args.end_run + 1):
                tasks.append((algo_name, scene, i, params, args.end_run))

    if not tasks:
        print("No new experiments to run. All results are up to date.")
        return

    # 单进程顺序执行实验
    for task in tasks:
        result = run_single_experiment(task)
        if result:
            print(result)

    print("\nAblation study runs complete.")

if __name__ == '__main__':
    main()
