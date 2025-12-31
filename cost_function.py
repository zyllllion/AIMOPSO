import numpy as np


def dist_p2s(p, a, b):
    p = np.array(p, dtype=float)
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    d = b - a
    dd = np.dot(d, d)
    if dd < 1e-12:
        return np.linalg.norm(p - a)
    t = np.clip(np.dot(p - a, d) / dd, 0, 1)
    proj = a + t * d
    return np.linalg.norm(p - proj)


def _compute_J4_like_matlab(x_all, y_all, z_abs):
    N = len(x_all)
    if N < 3: return 0.0
    J4_sum, n4 = 0.0, 0
    for i in range(N - 2):
        seg1, seg2 = None, None
        for j1 in range(i, -1, -1):
            v = np.array([x_all[j1 + 1] - x_all[j1], y_all[j1 + 1] - y_all[j1], z_abs[j1 + 1] - z_abs[j1]])
            if np.linalg.norm(v) > 1e-12: seg1 = v; break
        if seg1 is None: continue
        for j2 in range(i, N - 2):
            v = np.array([x_all[j2 + 2] - x_all[j2 + 1], y_all[j2 + 2] - y_all[j2 + 1], z_abs[j2 + 2] - z_abs[j2 + 1]])
            if np.linalg.norm(v) > 1e-12: seg2 = v; break
        if seg2 is None: continue

        n1, n2 = np.linalg.norm(seg1), np.linalg.norm(seg2)
        if n1 < 1e-12 or n2 < 1e-12: continue

        cosang = np.clip(np.dot(seg1, seg2) / (n1 * n2), -1.0, 1.0)
        ang = np.arccos(cosang)

        J4_sum += abs(ang) / np.pi
        n4 += 1

    return J4_sum / n4 if n4 > 0 else 0.0


def calculate_cost(sol_cartesian, model, terrain_interpolator, var_min):
    """
    Cost function - strictly aligned with MATLAB MyCost.m normalization logic
    """
    FINITE_PENALTY = np.inf

    x, y, z_rel = sol_cartesian['x'], sol_cartesian['y'], sol_cartesian['z']
    
    if np.any(np.isnan(x)) or np.any(np.isnan(y)) or np.any(np.isnan(z_rel)):
        return np.array([FINITE_PENALTY, FINITE_PENALTY, FINITE_PENALTY, FINITE_PENALTY])
    if np.any(np.isinf(x)) or np.any(np.isinf(y)) or np.any(np.isinf(z_rel)):
        return np.array([FINITE_PENALTY, FINITE_PENALTY, FINITE_PENALTY, FINITE_PENALTY])
    xs, ys, zs = model['start']
    xf, yf, zf = model['end']
    x_all = np.concatenate(([xs], x, [xf]))
    y_all = np.concatenate(([ys], y, [yf]))
    N = len(x_all)
    if terrain_interpolator is None: raise ValueError("calculate_cost requires a valid terrain_interpolator.")

    z_rel_all = np.concatenate(([zs], z_rel, [zf]))
    
    ground_z_all = np.zeros(N)
    for i in range(N):
        y_idx = int(np.round(y_all[i]))
        x_idx = int(np.round(x_all[i]))
        y_idx = np.clip(y_idx, 0, model['map_range'][1] - 1)
        x_idx = np.clip(x_idx, 0, model['map_range'][0] - 1)
        ground_z_all[i] = model['H'][y_idx, x_idx]
    
    z_abs_all = z_rel_all + ground_z_all

    z_abs = z_abs_all

    segments = np.diff(np.stack([x_all, y_all, z_abs], axis=1), axis=0)
    seg_lengths = np.linalg.norm(segments, axis=1)
    if np.any(seg_lengths <= var_min['r']):
        J1 = FINITE_PENALTY
    else:
        Traj = np.sum(seg_lengths)
        PP = np.linalg.norm(model['end'] - model['start'])
        J1 = abs(1 - PP / Traj) if Traj > 1e-9 else FINITE_PENALTY

    J2 = 0.0
    if 'threats' in model and len(model['threats']) > 0:
        threats, drone_size, danger_dist = model['threats'], 1, 10
        J2_sum = 0.0
        n2 = 0
        collision = False
        for threat in threats:
            tx, ty, _, R = threat
            for j in range(N - 1):
                d = dist_p2s([tx, ty], [x_all[j], y_all[j]], [x_all[j + 1], y_all[j + 1]])

                threat_cost = 0.0
                if d < (R + drone_size):
                    threat_cost = FINITE_PENALTY
                    collision = True
                elif d <= (R + drone_size + danger_dist):
                    threat_cost = 1 - (d - drone_size - R) / danger_dist

                J2_sum += threat_cost
                n2 += 1
            if collision: break

        if collision:
            J2 = FINITE_PENALTY
        else:
            J2 = J2_sum / n2 if n2 > 0 else 0.0

    zmax, zmin = model['zmax'], model['zmin']
    height_range = zmax - zmin
    z_middle = z_rel[:]
    
    if np.any(z_middle < 0):
        J3 = FINITE_PENALTY
    elif height_range < 1e-9:
        J3 = FINITE_PENALTY if np.any(np.abs(z_middle - zmin) > 1e-9) else 0.0
    else:
        normalized_z_dev = np.abs(z_middle - (zmax + zmin) / 2) / (height_range / 2)
        J3 = np.mean(normalized_z_dev)

    J4 = _compute_J4_like_matlab(x_all, y_all, z_abs)

    costs = np.array([J1, J2, J3, J4])
    costs[np.isnan(costs) | np.isinf(costs)] = FINITE_PENALTY
    return costs
