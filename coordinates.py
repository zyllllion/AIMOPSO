import numpy as np
def transformation_matrix(r, phi, psi):
    """
    4x4 homogeneous transformation matrix: T_i = R_z(φ_i) · R_y(-ψ_i) · T_x(r_i)
    
    Parameters:
    r: Step length (distance along current direction)
    phi: Azimuth angle (rotation around z-axis, deflection in horizontal plane)
    psi: Pitch angle (rotation around y-axis, vertical pitch, negative sign for right-hand coordinate system)
    
    Returns: 4x4 transformation matrix
    """
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    cos_psi, sin_psi = np.cos(-psi), np.sin(-psi)
    rot_z = np.array([
        [cos_phi, -sin_phi, 0, 0],
        [sin_phi,  cos_phi, 0, 0],
        [0,        0,       1, 0],
        [0,        0,       0, 1]
    ])
    rot_y = np.array([
        [ cos_psi, 0, sin_psi, 0],
        [ 0,       1, 0,       0],
        [-sin_psi, 0, cos_psi, 0],
        [ 0,       0, 0,       1]
    ])
    trans_x = np.array([
        [1, 0, 0, r],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return rot_z @ rot_y @ trans_x
def spherical_to_cartesian(solution, model):
    """
    Convert spherical coordinate path increments to (x,y,z) and clip to valid range.
    """
    r_vec = solution['r'][0]
    phi_vec = solution['phi'][0]
    psi_vec = solution['psi'][0]
    n = model['n']
    xs, ys, zs = model['start']
    start_transform = np.array([
        [1, 0, 0, xs],
        [0, 1, 0, ys],
        [0, 0, 1, zs],
        [0, 0, 0, 1]
    ])
    dir_vec = model['end'] - model['start']
    phi_start = np.arctan2(dir_vec[1], dir_vec[0])
    psi_start = np.arctan2(dir_vec[2], np.linalg.norm(dir_vec[:2]))
    start_pose = start_transform @ transformation_matrix(0, phi_start, psi_start)
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    T_cum = transformation_matrix(r_vec[0], phi_vec[0], psi_vec[0])
    pos = start_pose @ T_cum
    x[0], y[0], z[0] = pos[0, 3], pos[1, 3], pos[2, 3]
    for i in range(1, n):
        T_local = transformation_matrix(r_vec[i], phi_vec[i], psi_vec[i])
        T_cum = T_cum @ T_local
        pos = start_pose @ T_cum
        x[i], y[i], z[i] = pos[0, 3], pos[1, 3], pos[2, 3]
    x = np.clip(x, model['xmin'], model['xmax'])
    y = np.clip(y, model['ymin'], model['ymax'])
    z = np.clip(z, model['zmin'], model['zmax'])
    return {'x': x, 'y': y, 'z': z}

def cartesian_to_spherical(cartesian_path_rel, model):
    """
    Convert Cartesian coordinate path with relative height to spherical coordinates.
    
    Parameters:
    cartesian_path_rel: Nx3 array, z coordinate is relative height (height above ground)
    model: Environment model
    
    Key: Calculate relative angles, not absolute angles
    - phi, psi are angles relative to current movement direction
    - Step 1 is relative to initial direction (start -> end)
    - Step i is relative to direction of step i-1
    """
    points = np.vstack([model['start'], cartesian_path_rel])
    n_segments = len(cartesian_path_rel)
    
    r = np.zeros(n_segments)
    psi = np.zeros(n_segments)
    phi = np.zeros(n_segments)
    
    dir_vector = model['end'] - model['start']
    dir_xy = np.linalg.norm(dir_vector[:2])
    
    if dir_xy > 1e-9:
        initial_heading = np.arctan2(dir_vector[1], dir_vector[0])
        initial_elevation = np.arctan2(dir_vector[2], dir_xy)
    else:
        initial_heading = 0.0
        initial_elevation = np.pi / 2 if dir_vector[2] > 0 else -np.pi / 2
    
    prev_heading = initial_heading
    prev_elevation = initial_elevation
    
    for i in range(n_segments):
        segment = points[i + 1] - points[i]
        
        r[i] = np.linalg.norm(segment)
        
        if r[i] < 1e-9:
            psi[i] = 0.0
            phi[i] = 0.0
            continue
        
        segment_xy = np.linalg.norm(segment[:2])
        
        if segment_xy > 1e-9:
            current_heading = np.arctan2(segment[1], segment[0])
            current_elevation = np.arctan2(segment[2], segment_xy)
        else:
            current_heading = prev_heading
            current_elevation = np.pi / 2 if segment[2] > 0 else -np.pi / 2
        
        phi[i] = current_heading - prev_heading
        
        psi[i] = current_elevation - prev_elevation
        
        phi[i] = np.arctan2(np.sin(phi[i]), np.cos(phi[i]))
        psi[i] = np.arctan2(np.sin(psi[i]), np.cos(psi[i]))
        
        prev_heading = current_heading
        prev_elevation = current_elevation
    
    return {
        'r': r.reshape(1, -1),
        'psi': psi.reshape(1, -1),
        'phi': phi.reshape(1, -1)
    }
