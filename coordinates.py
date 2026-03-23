import numpy as np
def transformation_matrix(r, phi, psi):
    """
    4x4 齐次变换矩阵：T_i = R_z(φ_i) · R_y(-ψ_i) · T_x(r_i)
    
    参数:
    r: 步长 (沿当前方向的距离)
    phi: 方位角 (绕z轴旋转，水平面内的偏转)
    psi: 俯仰角 (绕y轴旋转，垂直方向的俯仰，使用负号符合右手坐标系)
    
    返回: 4x4变换矩阵
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
    将球坐标表示的路径增量转换为 (x,y,z)，并裁剪到合法范围（含 z 裁剪，保持与 MATLAB 一致）
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
    # ⚠️ 修复：z是相对高度，需要确保不违反地形约束
    # 不能简单裁剪到[zmin,zmax]，因为这可能导致穿越地形
    z = np.clip(z, model['zmin'], model['zmax'])
    return {'x': x, 'y': y, 'z': z}

def cartesian_to_spherical(cartesian_path_rel, model):
    """
    将相对高度的笛卡尔坐标路径转换为球坐标解。
    严格对齐MATLAB SphericalToCart2.m的逆运算。
    
    参数:
    cartesian_path_rel: Nx3数组，z坐标是相对高度（离地高度），与model['start']的z坐标定义一致
    model: 环境模型
    
    关键：计算相对转角，而非绝对角度
    - phi, psi 是相对于当前运动方向的转角
    - 第1步相对于初始方向（起点→终点）
    - 第i步相对于第i-1步的方向
    """
    points = np.vstack([model['start'], cartesian_path_rel])
    n_segments = len(cartesian_path_rel)
    
    # 初始化输出
    r = np.zeros(n_segments)
    psi = np.zeros(n_segments)
    phi = np.zeros(n_segments)
    
    # 计算初始方向（起点指向终点，对应MATLAB的phistart, psistart）
    dir_vector = model['end'] - model['start']
    dir_xy = np.linalg.norm(dir_vector[:2])
    
    if dir_xy > 1e-9:
        initial_heading = np.arctan2(dir_vector[1], dir_vector[0])  # 水平方位角
        initial_elevation = np.arctan2(dir_vector[2], dir_xy)       # 俯仰角
    else:
        initial_heading = 0.0
        initial_elevation = np.pi / 2 if dir_vector[2] > 0 else -np.pi / 2
    
    # 上一步的方向（初始为起点→终点的方向）
    prev_heading = initial_heading
    prev_elevation = initial_elevation
    
    for i in range(n_segments):
        # 当前航段向量
        segment = points[i + 1] - points[i]
        
        # r: 步长
        r[i] = np.linalg.norm(segment)
        
        if r[i] < 1e-9:
            # 如果两点重合，保持上一步的方向
            psi[i] = 0.0
            phi[i] = 0.0
            continue
        
        # 当前航段的全局方向
        segment_xy = np.linalg.norm(segment[:2])
        
        if segment_xy > 1e-9:
            current_heading = np.arctan2(segment[1], segment[0])
            current_elevation = np.arctan2(segment[2], segment_xy)
        else:
            # 垂直飞行
            current_heading = prev_heading
            current_elevation = np.pi / 2 if segment[2] > 0 else -np.pi / 2
        
        # 🔧 关键：计算相对转角（当前方向 - 上一步方向）
        # MATLAB定义: phi=水平方位角, psi=俯仰角
        # phi: 水平方位角的变化（azimuth/yaw）
        phi[i] = current_heading - prev_heading
        
        # psi: 俯仰角的变化（elevation/pitch）
        psi[i] = current_elevation - prev_elevation
        
        # 🔧 角度归一化到[-π, π]
        phi[i] = np.arctan2(np.sin(phi[i]), np.cos(phi[i]))
        psi[i] = np.arctan2(np.sin(psi[i]), np.cos(psi[i]))
        
        # 更新上一步方向
        prev_heading = current_heading
        prev_elevation = current_elevation
    
    # 将形状调整为 (1, n_var)
    return {
        'r': r.reshape(1, -1),
        'psi': psi.reshape(1, -1),
        'phi': phi.reshape(1, -1)
    }