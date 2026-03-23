"""
Analyze function-benchmark results produced by run_mo_function_benchmark.py.

Outputs:
- function_metrics_per_run.csv
- function_metrics_summary.csv
- pairwise_wilcoxon_symbols.csv
- plus_approx_minus_summary.csv
- friedman_avg_ranks.csv
- friedman_avg_ranks.png
"""

import argparse
import inspect
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, ranksums, wilcoxon

from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
from pymoo.problems import get_problem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions


DEFAULT_FUNCTIONS = ["dtlz1", "dtlz2", "dtlz3", "dtlz4", "dtlz7", "wfg1", "wfg2", "wfg3"]
DEFAULT_ALGORITHMS = [
    "IMOPSO-core",
    "MOPSO-core",
    "NSGA-II",
    "SPEA2",
    "SMS-EMOA",
    "NSGA-III",
    "RVEA",
    "AGEMOEA2",
]


def parse_list_arg(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def safe_alg_dir_name(name: str) -> str:
    return name.replace("*", "")


def get_problem_with_config(func_name: str, n_obj: int):
    name = func_name.lower()
    if name == "dtlz1":
        n_var = n_obj + 4
    elif name.startswith("dtlz"):
        n_var = n_obj + 9
    elif name.startswith("wfg"):
        n_var = 24
    else:
        raise ValueError(f"Unsupported function: {func_name}")
    return get_problem(name, n_var=n_var, n_obj=n_obj)


def load_runs(
    result_root: str,
    function_name: str,
    algorithm_name: str,
    start_run: int,
    end_run: int,
) -> Tuple[List[Optional[np.ndarray]], List[Optional[float]]]:
    run_data = []
    run_time = []
    alg_dir = os.path.join(result_root, f"function_{function_name.lower()}", safe_alg_dir_name(algorithm_name))
    for run_id in range(start_run, end_run + 1):
        csv_path = os.path.join(alg_dir, f"run_{run_id}.csv")
        meta_path = os.path.join(alg_dir, f"run_{run_id}_meta.json")

        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            df = pd.read_csv(csv_path)
            arr = df.values.astype(float)
            arr = arr[np.all(np.isfinite(arr), axis=1)]
            if len(arr) == 0:
                run_data.append(None)
            else:
                nd_idx = NonDominatedSorting().do(arr, only_non_dominated_front=True)
                run_data.append(arr[nd_idx])
        else:
            run_data.append(None)

        runtime = None
        if os.path.exists(meta_path) and os.path.getsize(meta_path) > 0:
            try:
                with open(meta_path, "r", encoding="utf-8") as fr:
                    meta = json.load(fr)
                runtime = float(meta.get("runtime_sec")) if meta.get("runtime_sec") is not None else None
            except Exception:
                runtime = None
        run_time.append(runtime)
    return run_data, run_time


def get_reference_pf(problem, n_obj: int) -> Optional[np.ndarray]:
    # Try to obtain a true/analytical PF from pymoo. If unavailable, return None.
    pf = None
    if not hasattr(problem, "pareto_front"):
        return None

    try:
        sig = inspect.signature(problem.pareto_front)
        if "ref_dirs" in sig.parameters:
            ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=15)
            pf = problem.pareto_front(ref_dirs=ref_dirs)
        else:
            pf = problem.pareto_front()
    except Exception:
        pf = None

    if pf is None:
        return None
    pf = np.array(pf, dtype=float)
    if pf.ndim == 1:
        pf = pf.reshape(1, -1)
    pf = pf[np.all(np.isfinite(pf), axis=1)]
    if len(pf) == 0:
        return None
    nd_idx = NonDominatedSorting().do(pf, only_non_dominated_front=True)
    return pf[nd_idx]


def fallback_union_pf(all_runs: Dict[str, List[Optional[np.ndarray]]]) -> Optional[np.ndarray]:
    stacks = []
    for runs in all_runs.values():
        for f in runs:
            if f is not None and len(f) > 0:
                stacks.append(f)
    if not stacks:
        return None
    F = np.vstack(stacks)
    nd_idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
    return F[nd_idx]


def normalize_front(F: np.ndarray, fmin: np.ndarray, frange: np.ndarray) -> np.ndarray:
    return (F - fmin) / frange


def build_hv_indicator(n_obj: int):
    ref_point = np.array([1.1] * n_obj)
    try:
        return Hypervolume(ref_point=ref_point, normalize=False)
    except TypeError:
        return Hypervolume(ref_point=ref_point)


def build_igd_indicator(ref_pf_norm: np.ndarray):
    # Compatible with different pymoo versions.
    try:
        return IGD(ref_pf_norm, zero_to_one=True)
    except TypeError:
        pass
    try:
        return IGD(ref_pf_norm)
    except TypeError:
        return IGD(pf=ref_pf_norm)


def indicator_value(indicator, F: np.ndarray) -> float:
    if hasattr(indicator, "do"):
        return float(indicator.do(F))
    return float(indicator(F))


def compare_symbol(
    ref_vals: List[float],
    cmp_vals: List[float],
    higher_is_better: bool,
    alpha: float = 0.05,
) -> Tuple[str, float]:
    n = min(len(ref_vals), len(cmp_vals))
    if n == 0:
        return "NA", np.nan

    ref_arr = np.array(ref_vals[:n], dtype=float)
    cmp_arr = np.array(cmp_vals[:n], dtype=float)
    ref_mean = float(np.mean(ref_arr))
    cmp_mean = float(np.mean(cmp_arr))

    # Prefer paired Wilcoxon signed-rank for aligned repeated runs.
    try:
        if np.any(np.abs(ref_arr - cmp_arr) > 1e-12):
            _, p_value = wilcoxon(ref_arr, cmp_arr, alternative="two-sided", zero_method="wilcox")
        else:
            p_value = 1.0
    except Exception:
        # Fallback to rank-sum if paired test fails.
        _, p_value = ranksums(ref_arr, cmp_arr)

    if p_value >= alpha:
        return "~", p_value
    if higher_is_better:
        return ("+" if ref_mean > cmp_mean else "-"), p_value
    return ("+" if ref_mean < cmp_mean else "-"), p_value


def main():
    parser = argparse.ArgumentParser(description="Analyze MO function benchmark results.")
    parser.add_argument("--functions", type=str, default=",".join(DEFAULT_FUNCTIONS))
    parser.add_argument("--algorithms", type=str, default=",".join(DEFAULT_ALGORITHMS))
    parser.add_argument("--reference_algorithm", type=str, default="IMOPSO-core")
    parser.add_argument("--n_obj", type=int, default=4)
    parser.add_argument("--start_run", type=int, default=1)
    parser.add_argument("--end_run", type=int, default=30)
    parser.add_argument("--result_root", type=str, default="run_results_mo_functions")
    parser.add_argument("--output_dir", type=str, default="analysis_mo_functions")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    functions = parse_list_arg(args.functions)
    algorithms = parse_list_arg(args.algorithms)
    os.makedirs(args.output_dir, exist_ok=True)

    all_per_run_rows = []
    all_summary_rows = []
    pairwise_rows = []

    friedman_hv_ranks = []
    friedman_igd_ranks = []

    # Aggregate +/~/- counts
    pam = {
        "HV": {alg: {"+": 0, "~": 0, "-": 0} for alg in algorithms if alg != args.reference_algorithm},
        "IGD": {alg: {"+": 0, "~": 0, "-": 0} for alg in algorithms if alg != args.reference_algorithm},
    }

    for func_name in functions:
        problem = get_problem_with_config(func_name, args.n_obj)
        runs_by_alg: Dict[str, List[Optional[np.ndarray]]] = {}
        runtimes_by_alg: Dict[str, List[Optional[float]]] = {}

        for alg in algorithms:
            runs, rt = load_runs(args.result_root, func_name, alg, args.start_run, args.end_run)
            runs_by_alg[alg] = runs
            runtimes_by_alg[alg] = rt

        ref_pf = get_reference_pf(problem, args.n_obj)
        if ref_pf is None:
            ref_pf = fallback_union_pf(runs_by_alg)
        if ref_pf is None or len(ref_pf) == 0:
            print(f"[WARN] {func_name}: no reference PF available, skip.")
            continue

        fmin = ref_pf.min(axis=0)
        fmax = ref_pf.max(axis=0)
        frange = np.where((fmax - fmin) < 1e-12, 1.0, fmax - fmin)
        ref_pf_norm = normalize_front(ref_pf, fmin, frange)

        hv_calc = build_hv_indicator(args.n_obj)
        igd_calc = build_igd_indicator(ref_pf_norm)

        metrics_for_friedman_hv = []
        metrics_for_friedman_igd = []
        metric_cache = {}

        for alg in algorithms:
            hv_vals = []
            igd_vals = []
            valid_runs = 0

            for idx, F in enumerate(runs_by_alg[alg], start=args.start_run):
                rt = runtimes_by_alg[alg][idx - args.start_run]
                if F is None or len(F) == 0:
                    continue
                nd_idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
                F_nd = F[nd_idx]
                F_norm = normalize_front(F_nd, fmin, frange)
                hv = indicator_value(hv_calc, F_norm)
                igd = indicator_value(igd_calc, F_norm)

                hv_vals.append(hv)
                igd_vals.append(igd)
                valid_runs += 1
                all_per_run_rows.append(
                    {
                        "function": func_name,
                        "algorithm": alg,
                        "run_id": idx,
                        "HV": hv,
                        "IGD": igd,
                        "runtime_sec": rt,
                        "n_points": int(len(F_nd)),
                    }
                )

            metric_cache[alg] = {"hv": hv_vals, "igd": igd_vals}
            hv_mean = float(np.mean(hv_vals)) if hv_vals else np.nan
            hv_std = float(np.std(hv_vals)) if hv_vals else np.nan
            igd_mean = float(np.mean(igd_vals)) if igd_vals else np.nan
            igd_std = float(np.std(igd_vals)) if igd_vals else np.nan
            rt_vals = [x for x in runtimes_by_alg[alg] if x is not None]
            rt_med = float(np.median(rt_vals)) if rt_vals else np.nan
            rt_iqr = float(np.percentile(rt_vals, 75) - np.percentile(rt_vals, 25)) if rt_vals else np.nan

            all_summary_rows.append(
                {
                    "function": func_name,
                    "algorithm": alg,
                    "valid_runs": valid_runs,
                    "HV_mean": hv_mean,
                    "HV_std": hv_std,
                    "IGD_mean": igd_mean,
                    "IGD_std": igd_std,
                    "runtime_median_sec": rt_med,
                    "runtime_iqr_sec": rt_iqr,
                }
            )

            metrics_for_friedman_hv.append(hv_mean)
            metrics_for_friedman_igd.append(igd_mean)

        # Pairwise vs reference
        if args.reference_algorithm in metric_cache:
            ref_hv = metric_cache[args.reference_algorithm]["hv"]
            ref_igd = metric_cache[args.reference_algorithm]["igd"]
            for alg in algorithms:
                if alg == args.reference_algorithm:
                    continue
                cmp_hv = metric_cache[alg]["hv"]
                cmp_igd = metric_cache[alg]["igd"]

                sym_hv, p_hv = compare_symbol(ref_hv, cmp_hv, higher_is_better=True, alpha=args.alpha)
                sym_igd, p_igd = compare_symbol(ref_igd, cmp_igd, higher_is_better=False, alpha=args.alpha)

                pairwise_rows.append(
                    {
                        "function": func_name,
                        "reference": args.reference_algorithm,
                        "compare": alg,
                        "HV_symbol": sym_hv,
                        "HV_pvalue": p_hv,
                        "IGD_symbol": sym_igd,
                        "IGD_pvalue": p_igd,
                    }
                )
                if sym_hv in pam["HV"][alg]:
                    pam["HV"][alg][sym_hv] += 1
                if sym_igd in pam["IGD"][alg]:
                    pam["IGD"][alg][sym_igd] += 1

        # Friedman rank with function-level means
        hv_arr = np.array(metrics_for_friedman_hv, dtype=float)
        igd_arr = np.array(metrics_for_friedman_igd, dtype=float)
        if np.all(np.isfinite(hv_arr)):
            # Higher HV is better -> rank with descending order
            friedman_hv_ranks.append(rankdata(-hv_arr, method="average"))
        if np.all(np.isfinite(igd_arr)):
            # Lower IGD is better
            friedman_igd_ranks.append(rankdata(igd_arr, method="average"))

    # Write outputs
    per_run_df = pd.DataFrame(all_per_run_rows)
    summary_df = pd.DataFrame(all_summary_rows)
    pairwise_df = pd.DataFrame(pairwise_rows)

    per_run_csv = os.path.join(args.output_dir, "function_metrics_per_run.csv")
    summary_csv = os.path.join(args.output_dir, "function_metrics_summary.csv")
    pairwise_csv = os.path.join(args.output_dir, "pairwise_wilcoxon_symbols.csv")
    per_run_df.to_csv(per_run_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    pairwise_df.to_csv(pairwise_csv, index=False)

    # +/~/- summary
    pam_rows = []
    for metric in ["HV", "IGD"]:
        for alg, c in pam[metric].items():
            pam_rows.append(
                {
                    "metric": metric,
                    "reference": args.reference_algorithm,
                    "compare": alg,
                    "+": c["+"],
                    "~": c["~"],
                    "-": c["-"],
                }
            )
    pam_df = pd.DataFrame(pam_rows)
    pam_csv = os.path.join(args.output_dir, "plus_approx_minus_summary.csv")
    pam_df.to_csv(pam_csv, index=False)

    # Friedman average ranks
    rank_rows = []
    hv_avg = None
    igd_avg = None
    if len(friedman_hv_ranks) > 0:
        hv_avg = np.mean(np.vstack(friedman_hv_ranks), axis=0)
    if len(friedman_igd_ranks) > 0:
        igd_avg = np.mean(np.vstack(friedman_igd_ranks), axis=0)

    for i, alg in enumerate(algorithms):
        rank_rows.append(
            {
                "algorithm": alg,
                "IGD_avg_rank": float(igd_avg[i]) if igd_avg is not None else np.nan,
                "HV_avg_rank": float(hv_avg[i]) if hv_avg is not None else np.nan,
            }
        )
    rank_df = pd.DataFrame(rank_rows)
    rank_csv = os.path.join(args.output_dir, "friedman_avg_ranks.csv")
    rank_df.to_csv(rank_csv, index=False)

    # Plot ranks
    if (hv_avg is not None) and (igd_avg is not None):
        x = np.arange(len(algorithms))
        width = 0.36
        plt.figure(figsize=(12, 5.5))
        plt.bar(x - width / 2, igd_avg, width=width, label="IGD (lower rank better)")
        plt.bar(x + width / 2, hv_avg, width=width, label="HV (lower rank better)")
        plt.xticks(x, algorithms, rotation=20, ha="right")
        plt.ylabel("Average Rank")
        plt.title("Friedman Average Ranks Across Benchmark Functions")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "friedman_avg_ranks.png"), dpi=300)
        plt.close()

    print("=" * 88)
    print("Analysis complete")
    print(f"per-run metrics : {per_run_csv}")
    print(f"summary metrics : {summary_csv}")
    print(f"pairwise symbols: {pairwise_csv}")
    print(f"+/~/- summary  : {pam_csv}")
    print(f"friedman ranks : {rank_csv}")
    print("=" * 88)


if __name__ == "__main__":
    main()
