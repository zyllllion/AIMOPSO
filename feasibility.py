import numpy as np


# Raw cost semantics used for reporting and filtering.
RAW_INFEASIBLE_PENALTY = np.inf

# Finite surrogate penalty only for optimizers that cannot handle inf.
PYMOO_INFEASIBLE_PENALTY = 1e6


def is_feasible_cost(cost_vec):
    """Shared feasibility rule for plotting, caching, and statistics."""
    if cost_vec is None:
        return False
    cost_arr = np.asarray(cost_vec, dtype=float)
    return np.all(np.isfinite(cost_arr)) and np.all(cost_arr < PYMOO_INFEASIBLE_PENALTY)
