import numpy as np
import heapq
from scipy.ndimage import zoom

def _heuristic(a, b):
    """计算两个点之间的欧几里得距离作为启发式函数。"""
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def _calculate_multi_objective_cost(current, neighbor, came_from, grid, threats_map, grid_size):
    """
    计算多目标代价函数
    
    综合考虑4个目标：
    - J1: 路径长度
    - J2: 威胁代价
    - J3: 地形高度代价
    - J4: 平滑度代价
    """
    # J1: 路径长度代价（欧几里得距离）
    distance_cost = _heuristic(current, neighbor)
    
    # J2: 威胁代价（从威胁地图读取）
    threat_cost = threats_map[neighbor[0], neighbor[1]] if threats_map is not None else 0
    
    # J3: 地形高度代价（归一化后的地形高度）
    terrain_cost = grid[neighbor[0], neighbor[1]]
    if terrain_cost == np.inf:
        return np.inf  # 障碍物
    
    # J4: 平滑度代价（转弯角度惩罚）
    smoothness_cost = 0
    if current in came_from:
        prev = came_from[current]
        # 计算转弯角度
        vec1 = np.array([current[0] - prev[0], current[1] - prev[1]])
        vec2 = np.array([neighbor[0] - current[0], neighbor[1] - current[1]])
        
        # 避免零向量
        if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
            # 计算夹角的余弦值 (-1到1)
            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            cos_angle = np.clip(cos_angle, -1, 1)
            # 转弯角度 (0到π)
            turn_angle = np.arccos(cos_angle)
            # 转弯代价：角度越大代价越高
            smoothness_cost = turn_angle / np.pi * 10  # 归一化到0-10
    
    # 多目标加权和
    # 权重可以根据实际需求调整
    # w1, w2, w3, w4 = 1.0, 3.0, 0.5, 0.8  # v1: 威胁权重过高，路径过长 (J1=0.211)
    # w1, w2, w3, w4 = 1.0, 1.5, 0.5, 1.0  # v2: J1改善但J3/J4恶化 (J1=0.072, J3=0.115, J4=0.121)
    # w1, w2, w3, w4 = 1.0, 1.5, 1.0, 1.5  # v3: 退化到v1，路径过长 (J1=0.211)
    w1, w2, w3, w4 = 1.0, 1.0, 1.0, 1.0  # v4: 完全均衡权重
    
    total_cost = (
        w1 * distance_cost +      # 路径长度
        w2 * threat_cost +         # 威胁代价（权重最高）
        w3 * terrain_cost +        # 地形代价
        w4 * smoothness_cost       # 平滑度代价
    )
    
    return total_cost


def _a_star_search(grid, start, goal, threats_map=None, grid_size=100):
    """在给定的网格上执行A*搜索。"""
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: _heuristic(start, goal)}
    oheap = []

    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        current = heapq.heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            data.append(start)
            return data[::-1]

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            
            # 边界检查
            if not (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]):
                continue
            
            # 使用多目标代价函数
            step_cost = _calculate_multi_objective_cost(
                current, neighbor, came_from, grid, threats_map, grid_size
            )
            
            # 如果是障碍物，跳过
            if step_cost == np.inf:
                continue
            
            tentative_g_score = gscore[current] + step_cost

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, np.inf):
                continue

            if tentative_g_score < gscore.get(neighbor, np.inf):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + _heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    return None # 未找到路径

def _create_cost_map(model, grid_size=100):
    """
    根据地形和威胁创建A*算法的成本地图。
    
    返回:
        terrain_map: 地形代价地图
        threats_map: 威胁代价地图（软约束）
    """
    H = model['H']
    map_range = model['map_range']
    threats = model.get('threats', np.array([]))

    # 1. 缩放地形图到指定的网格大小
    scale_y = grid_size / H.shape[0]
    scale_x = grid_size / H.shape[1]
    terrain_map = zoom(H, (scale_y, scale_x), order=1)

    # 2. 将地形高度转换为成本 (归一化到0-100)
    terrain_map = (terrain_map - np.min(terrain_map)) / (np.max(terrain_map) - np.min(terrain_map) + 1e-6) * 100

    # 3. 创建威胁代价地图（使用渐变惩罚而非硬约束）
    threats_map = np.zeros((grid_size, grid_size))
    x_scale = grid_size / map_range[0]
    y_scale = grid_size / map_range[1]

    for threat in threats:
        cx, cy, cz, r = threat[0], threat[1], threat[2], threat[3]
        grid_cx = int(cx * x_scale)
        grid_cy = int(cy * y_scale)
        grid_r = int(r * x_scale)
        
        # 创建距离场：离威胁中心越近代价越高
        for i in range(max(0, grid_cy - grid_r * 2), min(grid_size, grid_cy + grid_r * 2)):
            for j in range(max(0, grid_cx - grid_r * 2), min(grid_size, grid_cx + grid_r * 2)):
                dist = np.sqrt((i - grid_cy)**2 + (j - grid_cx)**2)
                
                if dist <= grid_r:
                    # 威胁核心区域：极高代价（接近障碍物）
                    terrain_map[i, j] = np.inf
                    threats_map[i, j] = np.inf
                elif dist <= grid_r * 1.5:
                    # 威胁边缘区域：高代价（软约束）
                    penalty = 100 * (1 - (dist - grid_r) / (grid_r * 0.5))
                    threats_map[i, j] = max(threats_map[i, j], penalty)

    return terrain_map, threats_map

def get_a_star_guidance_path(model, grid_size=150):
    """
    为HE-NMOPSO生成多目标A*引导路径。
    
    改进点：
    1. 综合考虑路径长度、威胁、地形、平滑度
    2. 使用渐变威胁惩罚而非硬约束
    3. 考虑转弯角度以提高平滑度
    4. 【优化】提升网格精度至150x150（原100x100），减少路径锯齿
    """
    print("    -> 使用多目标A*算法生成引导路径（精度150x150）...")
    
    # 创建地形和威胁代价地图
    terrain_map, threats_map = _create_cost_map(model, grid_size)
    map_range = model['map_range']

    # 将真实坐标转换为网格坐标
    start_x, start_y = model['start'][0], model['start'][1]
    end_x, end_y = model['end'][0], model['end'][1]

    start_grid = (int(start_y / map_range[1] * grid_size), int(start_x / map_range[0] * grid_size))
    goal_grid = (int(end_y / map_range[1] * grid_size), int(end_x / map_range[0] * grid_size))

    # 确保起止点在边界内
    start_grid = (max(0, min(grid_size - 1, start_grid[0])), max(0, min(grid_size - 1, start_grid[1])))
    goal_grid = (max(0, min(grid_size - 1, goal_grid[0])), max(0, min(grid_size - 1, goal_grid[1])))
    
    # 调试：打印网格坐标
    print(f"    [DEBUG A*] 起点世界坐标: ({start_x:.1f}, {start_y:.1f})")
    print(f"    [DEBUG A*] 终点世界坐标: ({end_x:.1f}, {end_y:.1f})")
    print(f"    [DEBUG A*] 起点网格坐标: {start_grid}")
    print(f"    [DEBUG A*] 终点网格坐标: {goal_grid}")

    # 使用多目标A*搜索
    path_grid = _a_star_search(terrain_map, start_grid, goal_grid, threats_map, grid_size)
    
    # 调试：打印原始网格路径
    if path_grid:
        print(f"    [DEBUG A*] 网格路径前3点: {path_grid[:3]}")
        print(f"    [DEBUG A*] 网格路径后3点: {path_grid[-3:]}")

    if path_grid is None:
        print("    -> 多目标A*未能找到路径，退回至简单A*...")
        # 降级：使用简单A*（只考虑地形）
        path_grid = _a_star_search(terrain_map, start_grid, goal_grid, None, grid_size)
        if path_grid is None:
            return None

    # 将网格路径转换回真实世界坐标
    path_world = []
    for y_grid, x_grid in path_grid:
        x_world = (x_grid / grid_size) * map_range[0]
        y_world = (y_grid / grid_size) * map_range[1]
        
        # 🔧 修正：避免路径点在地图边界上（会导致后续计算出错）
        # 将边界点稍微向内移动，确保在有效搜索空间内
        x_world = np.clip(x_world, 1, map_range[0] - 1)
        y_world = np.clip(y_world, 1, map_range[1] - 1)
        
        path_world.append([x_world, y_world])
    
    # 调试：打印世界坐标转换
    print(f"    [DEBUG A*] 原始路径长度: {len(path_grid)}")
    print(f"    [DEBUG A*] 世界坐标前3点: {path_world[:3]}")
    print(f"    [DEBUG A*] 世界坐标后3点: {path_world[-3:]}")

    # 对路径进行简化，只保留关键的转折点，以匹配model['n']的数量
    if len(path_world) > model['n'] + 2:
        indices = np.linspace(0, len(path_world) - 1, model['n'] + 2, dtype=int)
        simplified_path = [path_world[i] for i in indices]
        print(f"    [DEBUG A*] 简化路径，从{len(path_world)}点→{len(simplified_path)}点")
        print(f"    [DEBUG A*] 简化后前3点: {simplified_path[:3]}")
    else:
        simplified_path = path_world

    print(f"    -> 多目标A*成功生成路径（{len(simplified_path)}个航点）")
    
    # 返回中间点 (不包括起点和终点)
    final_path = np.array(simplified_path[1:-1])
    print(f"    [DEBUG A*] 最终返回路径（去掉首尾）: {len(final_path)}点")
    print(f"    [DEBUG A*] 最终返回前3点: {final_path[:3]}")
    return final_path
