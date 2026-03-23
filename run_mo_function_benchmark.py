"""
Run multi-objective benchmark functions for:
- IMOPSO-core (dual-leader PSO core, no A* guidance)
- MOPSO-core  (single-leader PSO core baseline)
- NSGA-II, SPEA2, SMS-EMOA, NSGA-III, RVEA, AGEMOEA2

Output structure:
run_results_mo_functions/
  function_<name>/
    <algorithm>/
      run_<k>.csv        # objective values (f1..fm)
      run_<k>_meta.json  # runtime, sizes, seed
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.problems import get_problem
from pymoo.termination import get_termination
from pymoo.optimize import minimize
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

CORE_ALGORITHMS = {"IMOPSO-core", "MOPSO-core"}


def dominates(cost1: np.ndarray, cost2: np.ndarray) -> bool:
    return np.all(cost1 <= cost2) and np.any(cost1 < cost2)


def crowding_distance(F: np.ndarray) -> np.ndarray:
    n_points, n_obj = F.shape
    if n_points <= 2:
        return np.full(n_points, np.inf)
    cd = np.zeros(n_points)
    for j in range(n_obj):
        idx = np.argsort(F[:, j])
        vals = F[idx, j]
        cd[idx[0]] = np.inf
        cd[idx[-1]] = np.inf
        denom = vals[-1] - vals[0]
        if abs(denom) < 1e-12:
            continue
        for i in range(1, n_points - 1):
            cd[idx[i]] += (vals[i + 1] - vals[i - 1]) / denom
    return np.nan_to_num(cd, nan=0.0, posinf=np.inf)


def truncate_by_nondom_and_crowding(
    F: np.ndarray, X: np.ndarray, n_keep: int
) -> Tuple[np.ndarray, np.ndarray]:
    if len(F) == 0:
        return F, X
    nds = NonDominatedSorting()
    fronts = nds.do(F, only_non_dominated_front=False)
    keep_idx = []
    for front in fronts:
        front = np.array(front, dtype=int)
        if len(keep_idx) + len(front) <= n_keep:
            keep_idx.extend(front.tolist())
        else:
            needed = n_keep - len(keep_idx)
            if needed <= 0:
                break
            cd = crowding_distance(F[front])
            order = np.argsort(-cd)  # descending
            keep_idx.extend(front[order[:needed]].tolist())
            break
    keep_idx = np.array(keep_idx, dtype=int)
    return F[keep_idx], X[keep_idx]


def evaluate_F(problem, X: np.ndarray) -> np.ndarray:
    F = problem.evaluate(X, return_values_of=["F"])
    F = np.array(F, dtype=float)
    if F.ndim == 1:
        F = F.reshape(1, -1)
    return F


def polynomial_mutation_vector(
    rng: np.random.Generator,
    x: np.ndarray,
    xl: np.ndarray,
    xu: np.ndarray,
    eta: float = 20.0,
    prob_var: float = 0.2,
) -> np.ndarray:
    y = x.copy()
    for i in range(y.shape[0]):
        if rng.random() > prob_var:
            continue
        if xu[i] - xl[i] < 1e-12:
            continue

        delta1 = (y[i] - xl[i]) / (xu[i] - xl[i])
        delta2 = (xu[i] - y[i]) / (xu[i] - xl[i])
        r = rng.random()
        mut_pow = 1.0 / (eta + 1.0)
        if r < 0.5:
            xy = 1.0 - delta1
            val = 2.0 * r + (1.0 - 2.0 * r) * (xy ** (eta + 1.0))
            delta_q = val**mut_pow - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * (xy ** (eta + 1.0))
            delta_q = 1.0 - val**mut_pow
        y[i] = y[i] + delta_q * (xu[i] - xl[i])
        y[i] = np.clip(y[i], xl[i], xu[i])
    return y


def select_tournament_leader(
    rng: np.random.Generator, rep_F: np.ndarray, rep_X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if len(rep_F) == 1:
        return rep_X[0], rep_F[0]
    i1, i2 = rng.integers(0, len(rep_F), size=2)
    f1, f2 = rep_F[i1], rep_F[i2]
    if dominates(f1, f2):
        return rep_X[i1], rep_F[i1]
    if dominates(f2, f1):
        return rep_X[i2], rep_F[i2]
    cd = crowding_distance(rep_F[[i1, i2]])
    if cd[0] > cd[1]:
        return rep_X[i1], rep_F[i1]
    if cd[1] > cd[0]:
        return rep_X[i2], rep_F[i2]
    pick = i1 if rng.random() < 0.5 else i2
    return rep_X[pick], rep_F[pick]


def select_global_best(rep_F: np.ndarray, rep_X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Normalize objective-wise, then choose smallest sum (minimization)
    fmin = rep_F.min(axis=0)
    fmax = rep_F.max(axis=0)
    den = np.where((fmax - fmin) < 1e-12, 1.0, fmax - fmin)
    norm = (rep_F - fmin) / den
    idx = int(np.argmin(norm.sum(axis=1)))
    return rep_X[idx], rep_F[idx]


def run_core_mopso(
    problem,
    seed: int,
    pop_size: int,
    n_gen: int,
    n_rep: int,
    dual_leader: bool,
    p_global: float = 0.7,
    w: float = 0.7,
    wdamp: float = 0.99,
    c1: float = 1.5,
    c2: float = 1.5,
    p_mut: float = 0.2,
    eta_mut: float = 20.0,
):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    n_var = problem.n_var
    xl = np.array(problem.xl, dtype=float).reshape(-1)
    xu = np.array(problem.xu, dtype=float).reshape(-1)
    if xl.size == 1:
        xl = np.full(n_var, xl.item())
    if xu.size == 1:
        xu = np.full(n_var, xu.item())

    X = rng.uniform(xl, xu, size=(pop_size, n_var))
    V = np.zeros((pop_size, n_var), dtype=float)
    Vmax = 0.5 * (xu - xl)
    Vmin = -Vmax

    F = evaluate_F(problem, X)
    fe_count = pop_size
    PbestX = X.copy()
    PbestF = F.copy()

    rep_F, rep_X = truncate_by_nondom_and_crowding(F, X, n_rep)
    if len(rep_F) == 0:
        rep_F, rep_X = F.copy(), X.copy()

    for _ in range(n_gen):
        if len(rep_F) == 0:
            rep_F, rep_X = truncate_by_nondom_and_crowding(F, X, n_rep)
            if len(rep_F) == 0:
                break

        gbest_x, _ = select_global_best(rep_F, rep_X)
        for i in range(pop_size):
            if dual_leader and rng.random() < p_global:
                leader_x = gbest_x
            else:
                leader_x, _ = select_tournament_leader(rng, rep_F, rep_X)

            r1 = rng.random(n_var)
            r2 = rng.random(n_var)
            V[i] = w * V[i] + c1 * r1 * (PbestX[i] - X[i]) + c2 * r2 * (leader_x - X[i])
            V[i] = np.clip(V[i], Vmin, Vmax)
            X[i] = X[i] + V[i]

            out = (X[i] < xl) | (X[i] > xu)
            V[i][out] *= -1.0
            X[i] = np.clip(X[i], xl, xu)

        # Polynomial mutation
        X_mut = np.array(
            [polynomial_mutation_vector(rng, X[i], xl, xu, eta=eta_mut, prob_var=p_mut) for i in range(pop_size)]
        )

        F = evaluate_F(problem, X)
        F_mut = evaluate_F(problem, X_mut)
        fe_count += 2 * pop_size

        # Pbest update
        for i in range(pop_size):
            if dominates(F[i], PbestF[i]):
                PbestF[i] = F[i].copy()
                PbestX[i] = X[i].copy()
            elif (not dominates(PbestF[i], F[i])) and (rng.random() < 0.5):
                PbestF[i] = F[i].copy()
                PbestX[i] = X[i].copy()

        all_F = np.vstack([rep_F, F, F_mut])
        all_X = np.vstack([rep_X, X, X_mut])
        rep_F, rep_X = truncate_by_nondom_and_crowding(all_F, all_X, n_rep)
        w *= wdamp

    return rep_X, rep_F, int(fe_count)


def parse_list_arg(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def safe_alg_dir_name(name: str) -> str:
    return name.replace("*", "")


def derive_n_gen_from_max_fe(max_fe: int, pop_size: int, is_core: bool) -> int:
    # Approximate FE accounting:
    # - Core PSO variants: init pop (pop) + per-gen (2 * pop) from X and X_mut evaluations
    # - Pymoo variants (n_offsprings=pop defaults): total FE ~= n_gen * pop
    if is_core:
        if max_fe <= pop_size:
            return 1
        n_gen = int((max_fe - pop_size) // (2 * pop_size))
        return max(1, n_gen)
    n_gen = int(max_fe // pop_size)
    return max(1, n_gen)


def expected_fe(pop_size: int, n_gen: int, is_core: bool) -> int:
    if is_core:
        return int(pop_size + 2 * pop_size * n_gen)
    return int(pop_size * n_gen)


def get_problem_with_config(func_name: str, n_obj: int):
    name = func_name.lower()
    if name == "dtlz1":
        n_var = n_obj + 4  # k=5
    elif name.startswith("dtlz"):
        n_var = n_obj + 9  # k=10
    elif name.startswith("wfg"):
        n_var = 24
    else:
        raise ValueError(f"Unsupported function: {func_name}")
    return get_problem(name, n_var=n_var, n_obj=n_obj)


def run_one_pymoo(alg_name, problem, seed, pop_size, n_gen, ref_dirs):
    termination = get_termination("n_gen", n_gen)
    if alg_name == "NSGA-II":
        algo = NSGA2(pop_size=pop_size)
    elif alg_name == "SPEA2":
        algo = SPEA2(pop_size=pop_size)
    elif alg_name == "SMS-EMOA":
        algo = SMSEMOA(pop_size=pop_size)
    elif alg_name == "NSGA-III":
        algo = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)
    elif alg_name == "RVEA":
        algo = RVEA(pop_size=pop_size, ref_dirs=ref_dirs)
    elif alg_name == "AGEMOEA2":
        algo = AGEMOEA2(pop_size=pop_size)
    else:
        raise ValueError(f"Unsupported Pymoo algorithm: {alg_name}")

    res = minimize(problem, algo, termination, seed=seed, verbose=False)
    approx_n_off = int(getattr(algo, "n_offsprings", pop_size) or pop_size)
    fe_count = int(approx_n_off * n_gen)
    try:
        if res is not None and getattr(res, "algorithm", None) is not None:
            n_eval = getattr(res.algorithm.evaluator, "n_eval", None)
            if n_eval is not None:
                fe_count = int(n_eval)
    except Exception:
        pass
    if res is None or res.F is None:
        return np.empty((0, problem.n_var)), np.empty((0, problem.n_obj)), fe_count
    F = np.array(res.F, dtype=float)
    X = np.array(res.X, dtype=float) if res.X is not None else np.empty((len(F), problem.n_var))
    if F.ndim == 1:
        F = F.reshape(1, -1)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X, F, fe_count


def main():
    parser = argparse.ArgumentParser(description="Run MO benchmark functions with unified protocol.")
    parser.add_argument("--functions", type=str, default=",".join(DEFAULT_FUNCTIONS))
    parser.add_argument("--algorithms", type=str, default=",".join(DEFAULT_ALGORITHMS))
    parser.add_argument("--n_obj", type=int, default=4)
    parser.add_argument("--pop_size", type=int, default=120)
    parser.add_argument("--n_gen", type=int, default=500)
    parser.add_argument("--n_rep", type=int, default=50)
    parser.add_argument(
        "--max_fe",
        type=int,
        default=None,
        help="Fair mode FE budget per run. If set, n_gen is derived per algorithm to align FE.",
    )
    parser.add_argument(
        "--n_partitions",
        type=int,
        default=7,
        help="Das-Dennis partitions for reference directions (NSGA-III/RVEA).",
    )
    parser.add_argument("--start_run", type=int, default=1)
    parser.add_argument("--end_run", type=int, default=30)
    parser.add_argument("--output_root", type=str, default="run_results_mo_functions")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    functions = parse_list_arg(args.functions)
    algorithms = parse_list_arg(args.algorithms)

    ref_dirs = get_reference_directions("das-dennis", args.n_obj, n_partitions=args.n_partitions)
    os.makedirs(args.output_root, exist_ok=True)

    if args.pop_size < len(ref_dirs):
        print(
            f"[WARN] pop_size={args.pop_size} < ref_dirs={len(ref_dirs)}. "
            "For strict fairness with NSGA-III/RVEA, use pop_size >= ref_dirs."
        )

    print("=" * 88)
    print("MO Function Benchmark Runner")
    print(f"Functions : {functions}")
    print(f"Algorithms: {algorithms}")
    print(f"Runs      : {args.start_run}..{args.end_run}")
    if args.max_fe is None:
        print(f"Budget    : pop={args.pop_size}, gen={args.n_gen} (uniform generations)")
    else:
        print(f"Budget    : pop={args.pop_size}, max_fe={args.max_fe} (FE-fair mode)")
    print(f"RefDirs   : {len(ref_dirs)} (n_partitions={args.n_partitions})")
    print("=" * 88)

    for func_name in functions:
        problem = get_problem_with_config(func_name, args.n_obj)
        print(f"\n[Function] {func_name} (n_var={problem.n_var}, n_obj={problem.n_obj})")

        for alg_name in algorithms:
            safe_name = safe_alg_dir_name(alg_name)
            out_dir = os.path.join(args.output_root, f"function_{func_name.lower()}", safe_name)
            os.makedirs(out_dir, exist_ok=True)

            print(f"  [Algorithm] {alg_name}")
            is_core = alg_name in CORE_ALGORITHMS
            n_gen_eff = (
                args.n_gen
                if args.max_fe is None
                else derive_n_gen_from_max_fe(args.max_fe, args.pop_size, is_core=is_core)
            )
            fe_expected = expected_fe(args.pop_size, n_gen_eff, is_core=is_core)
            if args.max_fe is not None and fe_expected != args.max_fe:
                print(
                    f"    [note] target_fe={args.max_fe}, achievable_fe={fe_expected} "
                    f"(discrete budget step with pop={args.pop_size})"
                )

            for run_id in range(args.start_run, args.end_run + 1):
                out_csv = os.path.join(out_dir, f"run_{run_id}.csv")
                out_meta = os.path.join(out_dir, f"run_{run_id}_meta.json")
                if args.skip_existing and os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
                    print(f"    - run {run_id}: skip (exists)")
                    continue

                t0 = time.time()
                try:
                    if alg_name == "IMOPSO-core":
                        X, F, fe_actual = run_core_mopso(
                            problem,
                            seed=run_id,
                            pop_size=args.pop_size,
                            n_gen=n_gen_eff,
                            n_rep=args.n_rep,
                            dual_leader=True,
                        )
                    elif alg_name == "MOPSO-core":
                        X, F, fe_actual = run_core_mopso(
                            problem,
                            seed=run_id,
                            pop_size=args.pop_size,
                            n_gen=n_gen_eff,
                            n_rep=args.n_rep,
                            dual_leader=False,
                        )
                    else:
                        X, F, fe_actual = run_one_pymoo(
                            alg_name, problem, run_id, args.pop_size, n_gen_eff, ref_dirs
                        )

                    # Final unified truncation by non-domination + crowding
                    if len(F) > 0:
                        finite = np.all(np.isfinite(F), axis=1)
                        F = F[finite]
                        X = X[finite] if len(X) == len(finite) else X
                    if len(F) > 0:
                        F, X = truncate_by_nondom_and_crowding(F, X, args.n_rep)

                    elapsed = time.time() - t0
                    if len(F) > 0:
                        cols = [f"f{i+1}" for i in range(F.shape[1])]
                        pd.DataFrame(F, columns=cols).to_csv(out_csv, index=False)
                    else:
                        open(out_csv, "w", encoding="utf-8").close()

                    meta = {
                        "function": func_name,
                        "algorithm": alg_name,
                        "run_id": run_id,
                        "seed": run_id,
                        "runtime_sec": float(elapsed),
                        "n_solutions": int(len(F)),
                        "pop_size": args.pop_size,
                        "n_gen": int(n_gen_eff),
                        "n_obj": args.n_obj,
                        "max_fe_target": int(args.max_fe) if args.max_fe is not None else None,
                        "fe_expected": int(fe_expected),
                        "fe_actual": int(fe_actual),
                    }
                    with open(out_meta, "w", encoding="utf-8") as fw:
                        json.dump(meta, fw, ensure_ascii=False, indent=2)
                    print(
                        f"    - run {run_id}: ok ({elapsed:.2f}s, n={len(F)}, "
                        f"gen={n_gen_eff}, fe={fe_actual})"
                    )
                except Exception as e:
                    elapsed = time.time() - t0
                    open(out_csv, "w", encoding="utf-8").close()
                    with open(out_meta, "w", encoding="utf-8") as fw:
                        json.dump(
                            {
                                "function": func_name,
                                "algorithm": alg_name,
                                "run_id": run_id,
                                "seed": run_id,
                                "runtime_sec": float(elapsed),
                                "max_fe_target": int(args.max_fe) if args.max_fe is not None else None,
                                "error": str(e),
                            },
                            fw,
                            ensure_ascii=False,
                            indent=2,
                        )
                    print(f"    - run {run_id}: error -> {e}")

    print("\nDone. Results written to:", args.output_root)


if __name__ == "__main__":
    main()
