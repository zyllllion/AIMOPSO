# 📌 使用示例：7个算法全面对比
# python run_full_mo_benchmark2.py --scene 2 --algorithm A*IMOPSO
# python run_full_mo_benchmark2.py --scene 2 --algorithm NSGA-II
# python run_full_mo_benchmark2.py --scene 4 --algorithm SPEA2 --start_run 1 --end_run 30
# python run_full_mo_benchmark2.py --scene 2 --algorithm SMS-EMOA
# python run_full_mo_benchmark2.py --scene 2 --algorithm NSGA-III
# python run_full_mo_benchmark2.py --scene 2 --algorithm RVEA
# python run_full_mo_benchmark2.py --scene 2 --algorithm AGEMOEA2
# python run_full_mo_benchmark2.py --scene 3 --algorithm RVEA --start_run 29 --end_run 29

import numpy as np
import time
import os
import sys
import json
import pandas as pd
import argparse
from scipy.interpolate import RegularGridInterpolator

# Avoid console encoding crashes on non-UTF8 terminals (e.g., GBK on Windows).
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# --- Pymoo Imports ---
try:
    from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowding
    from pymoo.algorithms.moo.spea2 import SPEA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.algorithms.moo.sms import SMSEMOA
    from pymoo.algorithms.moo.rvea import RVEA
    from pymoo.algorithms.moo.age2 import AGEMOEA2
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    from pymoo.core.problem import Problem
    from pymoo.core.population import Population
    from pymoo.util.ref_dirs import get_reference_directions
except ImportError as e:
    print(f"[ERROR] Failed to import Pymoo: {e}", file=sys.stderr)
    sys.exit(1)

# --- Custom Imports ---
try:
    from aimopso_runner import run_aimopso
    from environments import create_scene1_model, create_scene2_model, create_scene3_model, create_scene4_model
    from cost_function import calculate_cost
except ImportError as e:
    print(f"[ERROR] Failed to import custom modules: {e}", file=sys.stderr)
    sys.exit(1)


def run_aimopso_for_benchmark(model, seed, max_it, n_pop, n_rep):
    """ 
    A wrapper for the A*IMOPSO algorithm to ensure the full repository object is returned
    for benchmark analysis.
    """
    _, final_rep, _ = run_aimopso(
        model,
        seed=seed,
        mode='full',
        max_it_override=max_it,
        n_pop_override=n_pop,
        n_rep_override=n_rep,
    )
    fe_actual = int(n_pop + 2 * n_pop * max_it)
    return final_rep, fe_actual


def derive_n_gen_from_max_fe(max_fe, pop_size, is_core):
    if is_core:
        if max_fe <= pop_size:
            return 1
        n_gen = int((max_fe - pop_size) // (2 * pop_size))
        return max(1, n_gen)
    n_gen = int(max_fe // pop_size)
    return max(1, n_gen)


def expected_fe(pop_size, n_gen, is_core):
    if is_core:
        return int(pop_size + 2 * pop_size * n_gen)
    return int(pop_size * n_gen)


class PymooUAVProblem(Problem):
    """Pymoo problem definition for UAV path planning."""

    def __init__(self, model):
        self.model = model
        self.terrain_interpolator = model['terrain_interpolator']
        self.var_min_for_cost = {'r': 3 * np.linalg.norm(model['start'] - model['end']) / model['n'] / 9}
        n_wp = self.model['n']

        xl = np.tile([self.model['xmin'], self.model['ymin'], self.model['zmin']], n_wp)
        xu = np.tile([self.model['xmax'], self.model['ymax'], self.model['zmax']], n_wp)

        # ⚠️ 关键修复：使用约束而不是惩罚值
        # n_ieq_constr=1 表示1个不等式约束（可行性检查）
        super().__init__(n_var=3 * n_wp, n_obj=4, n_ieq_constr=1, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        FEAS_THRESHOLD = 1e5  # 可行性阈值
        all_costs = []
        all_constraints = []
        
        for individual in X:
            waypoints = individual.reshape(self.model['n'], 3)
            sol_cartesian = {'x': waypoints[:, 0], 'y': waypoints[:, 1], 'z': waypoints[:, 2]}
            costs = calculate_cost(sol_cartesian, self.model, self.terrain_interpolator, self.var_min_for_cost)
            
            # 约束值：如果任何目标 >= FEAS_THRESHOLD，则违反约束
            # Pymoo约束：g(x) <= 0 为可行
            constraint_violation = np.max(costs) - FEAS_THRESHOLD  # 如果max(costs) >= 1e5，则 > 0（违反）
            
            all_costs.append(costs)
            all_constraints.append([constraint_violation])
        
        out["F"] = np.array(all_costs)
        out["G"] = np.array(all_constraints)  # 约束值


def main():
    """Main function to run the benchmark for a specified scene and algorithm."""
    parser = argparse.ArgumentParser(description="Run a specific MOO algorithm for a specific scene.")
    parser.add_argument('--scene', type=int, required=True, choices=[1, 2, 3, 4], help='Scene to run (1, 2, 3, or 4)')
    parser.add_argument('--algorithm', type=str, required=True, 
                        choices=['A*IMOPSO', 'NSGA-II', 'SPEA2', 'SMS-EMOA', 'NSGA-III', 'RVEA', 'AGEMOEA2'],
                        help='Algorithm to run')
    parser.add_argument('--start_run', type=int, default=1, help='The starting run number (inclusive). Defaults to 1.')
    parser.add_argument('--end_run', type=int, default=30, help='The ending run number (inclusive). Defaults to 30.')
    parser.add_argument('--pop_size', type=int, default=120, help='Population size for all algorithms.')
    parser.add_argument('--n_gen', type=int, default=500, help='Base generations/iterations when max_fe is not set.')
    parser.add_argument('--n_rep', type=int, default=50, help='Archive/kept solution size.')
    parser.add_argument('--max_fe', type=int, default=None, help='FE budget per run for strict fairness.')
    parser.add_argument('--n_partitions', type=int, default=7, help='Das-Dennis partitions for ref_dirs.')
    parser.add_argument('--results_root_dir', type=str, default='run_results_mo_sota_fair',
                        help='Output root directory.')
    args = parser.parse_args()

    # --- Benchmark Configuration ---
    SCENE_TO_RUN = args.scene
    ALGORITHM_TO_RUN = args.algorithm
    START_RUN = args.start_run
    END_RUN = args.end_run
    POP_SIZE = args.pop_size
    N_GEN = args.n_gen
    N_REP = args.n_rep
    MAX_FE = args.max_fe
    RESULTS_ROOT_DIR = args.results_root_dir
    # -----------------------------

    print("=" * 80)
    print(f"🚀 SOTA Algorithm Benchmark: SCENE {SCENE_TO_RUN}, ALGORITHM {ALGORITHM_TO_RUN}")
    if MAX_FE is None:
        print(f"Budget mode: pop={POP_SIZE}, n_gen={N_GEN} (uniform generations)")
    else:
        print(f"Budget mode: pop={POP_SIZE}, max_fe={MAX_FE} (strict FE fairness)")
    print("=" * 80)

    # 1. Create Environment
    print(f"Creating environment for Scene {SCENE_TO_RUN}...")
    scene_creators = {1: create_scene1_model, 2: create_scene2_model, 3: create_scene3_model, 4: create_scene4_model}
    model = scene_creators[SCENE_TO_RUN]()
    model['terrain_interpolator'] = RegularGridInterpolator(
        (np.arange(model['map_range'][1]), np.arange(model['map_range'][0])), model['H'],
        bounds_error=False, fill_value=0
    )
    print(f"--- Environment '{model.get('name', 'Unknown')}' loaded ---")

    # 2. Define Algorithms
    pymoo_problem = PymooUAVProblem(model)
    
    # 🎯 为NSGA-III和RVEA生成参考方向（4目标问题）
    ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=args.n_partitions)
    print(f"📐 生成参考方向数量: {len(ref_dirs)} (用于NSGA-III和RVEA)")
    if POP_SIZE < len(ref_dirs):
        print(
            f"⚠️  警告: pop_size={POP_SIZE} < ref_dirs={len(ref_dirs)}，"
            f"建议将 pop_size 调整到至少 {len(ref_dirs)}。"
        )

    is_core_algorithm = (ALGORITHM_TO_RUN == "A*IMOPSO")
    n_gen_eff = (
        N_GEN if MAX_FE is None
        else derive_n_gen_from_max_fe(MAX_FE, POP_SIZE, is_core=is_core_algorithm)
    )
    fe_expected = expected_fe(POP_SIZE, n_gen_eff, is_core=is_core_algorithm)
    if MAX_FE is not None and fe_expected != MAX_FE:
        print(
            f"[note] target_fe={MAX_FE}, achievable_fe={fe_expected} "
            f"(discrete budget step with pop={POP_SIZE})"
        )

    def run_pymoo_for_benchmark(alg_name, seed):
        termination = get_termination("n_gen", n_gen_eff)
        if alg_name == "NSGA-II":
            algo = NSGA2(pop_size=POP_SIZE)
        elif alg_name == "SPEA2":
            algo = SPEA2(pop_size=POP_SIZE)
        elif alg_name == "SMS-EMOA":
            algo = SMSEMOA(pop_size=POP_SIZE)
        elif alg_name == "NSGA-III":
            algo = NSGA3(pop_size=POP_SIZE, ref_dirs=ref_dirs)
        elif alg_name == "RVEA":
            algo = RVEA(pop_size=POP_SIZE, ref_dirs=ref_dirs)
        elif alg_name == "AGEMOEA2":
            algo = AGEMOEA2(pop_size=POP_SIZE)
        else:
            raise ValueError(f"Unknown algorithm: {alg_name}")

        res = minimize(pymoo_problem, algo, termination, seed=seed, verbose=False)
        n_off = int(getattr(algo, "n_offsprings", POP_SIZE) or POP_SIZE)
        fe_actual = int(n_off * n_gen_eff)
        try:
            if res is not None and getattr(res, "algorithm", None) is not None:
                n_eval = getattr(res.algorithm.evaluator, "n_eval", None)
                if n_eval is not None:
                    fe_actual = int(n_eval)
        except Exception:
            pass
        return res, fe_actual

    # Print algorithm information
    alg_info = {
        "A*IMOPSO": "2024 - Your algorithm: PSO + A* guidance + Dual leaders",
        "NSGA-II": "2002 - Non-dominated Sorting GA II (Fast sorting, crowding distance)",
        "SPEA2": "2001 - Strength Pareto EA 2 (External archive, k-NN density)",
        "SMS-EMOA": "2007 - S-Metric Selection EMOA (Hypervolume-driven)",
        "NSGA-III": "2014 - Non-dominated Sorting GA III (Many-objective, ref points)",
        "RVEA": "2016 - Reference Vector guided EA (Adaptive vectors)",
        "AGEMOEA2": "2022 - Adaptive Geometry Estimation II (Latest SOTA)"
    }
    print(f"\n📊 Algorithm: {ALGORITHM_TO_RUN}")
    print(f"📝 Description: {alg_info[ALGORITHM_TO_RUN]}")

    # 3. Run Benchmark Loop for the specified algorithm
    alg_name = ALGORITHM_TO_RUN
    print(f"⚖️  Effective budget: n_gen/max_it={n_gen_eff}, expected_fe={fe_expected}")

    print(f"\n--- Running Algorithm: {alg_name} ---")
    
    # 目录名映射：将算法名转换为Windows合法的目录名（Windows不支持*字符）
    dir_name_mapping = {
        "A*IMOPSO": "AIMOPSO",
        "NSGA-II": "NSGA-II",
        "SPEA2": "SPEA2",
        "SMS-EMOA": "SMS-EMOA",
        "NSGA-III": "NSGA-III",
        "RVEA": "RVEA",
        "AGEMOEA2": "AGEMOEA2"
    }
    safe_dir_name = dir_name_mapping.get(alg_name, alg_name)
    output_dir = os.path.join(RESULTS_ROOT_DIR, f"scene_{SCENE_TO_RUN}", safe_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    for i in range(START_RUN, END_RUN + 1):
        # Check if the result file already exists and is not empty
        result_file_path = os.path.join(output_dir, f"run_{i}.csv")
        meta_file_path = os.path.join(output_dir, f"run_{i}_meta.json")
        if os.path.exists(result_file_path) and os.path.getsize(result_file_path) > 0:
            print(f"  ⏭️  Skipping Run {i}/{END_RUN} (already exists)")
            continue

        start_time = time.time()
        print(f"  🔄 Run {i}/{END_RUN}...", end="", flush=True)

        current_seed = i

        try:
            if alg_name == "A*IMOPSO":
                final_rep, fe_actual = run_aimopso_for_benchmark(
                    model, seed=current_seed, max_it=n_gen_eff, n_pop=POP_SIZE, n_rep=N_REP
                )
                result_costs = np.array([p['Cost'] for p in final_rep]) if final_rep else np.array([])
            else:  # Pymoo SOTA algorithms
                result, fe_actual = run_pymoo_for_benchmark(alg_name, seed=current_seed)

                if result and result.opt is not None:
                    # Keep a unified final set size for metric stability.
                    if len(result.opt) > N_REP:
                        # Create a new population object just for the optimal set to be ranked
                        opt_pop = Population.new("X", result.opt.get("X"))
                        opt_pop.set("F", result.opt.get("F"))

                        survival = RankAndCrowding()
                        survivors = survival.do(pymoo_problem, opt_pop, n_survive=N_REP)

                        result_costs = survivors.get("F")

                    else:
                        result_costs = result.F
                else:
                    result_costs = np.array([])

            if result_costs.size > 0:
                # 注：Pymoo约束机制应该已经过滤了不可行解，但作为双保险再检查一次
                FEAS_THRESHOLD = 1e5
                df = pd.DataFrame(result_costs, columns=[f"J{j + 1}" for j in range(result_costs.shape[1])])
                feasible_df = df[(df < FEAS_THRESHOLD).all(axis=1)]
                
                if len(feasible_df) > 0:
                    feasible_df.to_csv(os.path.join(output_dir, f"run_{i}.csv"), index=False)
                    meta = {
                        "scene": SCENE_TO_RUN,
                        "algorithm": alg_name,
                        "run_id": i,
                        "seed": current_seed,
                        "runtime_sec": float(time.time() - start_time),
                        "pop_size": POP_SIZE,
                        "n_rep": N_REP,
                        "n_gen_or_max_it": int(n_gen_eff),
                        "max_fe_target": int(MAX_FE) if MAX_FE is not None else None,
                        "fe_expected": int(fe_expected),
                        "fe_actual": int(fe_actual),
                        "n_solutions": int(len(feasible_df)),
                    }
                    with open(meta_file_path, "w", encoding="utf-8") as fw:
                        json.dump(meta, fw, ensure_ascii=False, indent=2)
                    if len(feasible_df) < len(df):
                        print(
                            f" ✅ Done ({time.time() - start_time:.2f}s, "
                            f"{len(feasible_df)}/{len(df)} feasible, fe={fe_actual})"
                        )
                    else:
                        print(
                            f" ✅ Done ({time.time() - start_time:.2f}s, "
                            f"{len(feasible_df)} solutions, fe={fe_actual})"
                        )
                else:
                    open(os.path.join(output_dir, f"run_{i}.csv"), 'w').close()
                    with open(meta_file_path, "w", encoding="utf-8") as fw:
                        json.dump(
                            {
                                "scene": SCENE_TO_RUN,
                                "algorithm": alg_name,
                                "run_id": i,
                                "seed": current_seed,
                                "runtime_sec": float(time.time() - start_time),
                                "pop_size": POP_SIZE,
                                "n_rep": N_REP,
                                "n_gen_or_max_it": int(n_gen_eff),
                                "max_fe_target": int(MAX_FE) if MAX_FE is not None else None,
                                "fe_expected": int(fe_expected),
                                "fe_actual": int(fe_actual),
                                "n_solutions": 0,
                            },
                            fw,
                            ensure_ascii=False,
                            indent=2,
                        )
                    print(f" ⚠️  Done ({time.time() - start_time:.2f}s, 0/{len(df)} feasible, fe={fe_actual})")
            else:
                open(os.path.join(output_dir, f"run_{i}.csv"), 'w').close()
                with open(meta_file_path, "w", encoding="utf-8") as fw:
                    json.dump(
                        {
                            "scene": SCENE_TO_RUN,
                            "algorithm": alg_name,
                            "run_id": i,
                            "seed": current_seed,
                            "runtime_sec": float(time.time() - start_time),
                            "pop_size": POP_SIZE,
                            "n_rep": N_REP,
                            "n_gen_or_max_it": int(n_gen_eff),
                            "max_fe_target": int(MAX_FE) if MAX_FE is not None else None,
                            "fe_expected": int(fe_expected),
                            "fe_actual": int(fe_actual),
                            "n_solutions": 0,
                        },
                        fw,
                        ensure_ascii=False,
                        indent=2,
                    )
                print(f" ⚠️  Done ({time.time() - start_time:.2f}s, No solutions found, fe={fe_actual})")

        except Exception as e:
            print(f"\n❌ [ERROR] An exception occurred during run {i} of {alg_name}: {e}", file=sys.stderr)
            open(os.path.join(output_dir, f"run_{i}.csv"), 'w').close()
            with open(meta_file_path, "w", encoding="utf-8") as fw:
                json.dump(
                    {
                        "scene": SCENE_TO_RUN,
                        "algorithm": alg_name,
                        "run_id": i,
                        "seed": current_seed,
                        "runtime_sec": float(time.time() - start_time),
                        "pop_size": POP_SIZE,
                        "n_rep": N_REP,
                        "n_gen_or_max_it": int(n_gen_eff),
                        "max_fe_target": int(MAX_FE) if MAX_FE is not None else None,
                        "fe_expected": int(fe_expected),
                        "error": str(e),
                    },
                    fw,
                    ensure_ascii=False,
                    indent=2,
                )

    print("\n" + "=" * 80)
    print(f"🎉 SOTA Benchmark Data Generation for SCENE {SCENE_TO_RUN} COMPLETE.")
    print(f"📁 Results saved to: {output_dir}")
    print("🔄 You can now run the analysis script.")
    print("=" * 80)


if __name__ == "__main__":
    main()
