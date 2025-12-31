import numpy as np
import heapq
from scipy.ndimage import zoom

def _heuristic(a, b):
    """Calculate Euclidean distance between two points as heuristic function."""
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def _calculate_multi_objective_cost(current, neighbor, came_from, grid, threats_map, grid_size):
    """
    Calculate multi-objective cost: path length + threats + terrain + smoothness.
    """
    distance_cost = _heuristic(current, neighbor)
    threat_cost = threats_map[neighbor[0], neighbor[1]] if threats_map is not None else 0
    terrain_cost = grid[neighbor[0], neighbor[1]]
    
    if terrain_cost == np.inf:
        return np.inf
    
    smoothness_cost = 0
    if current in came_from:
        prev = came_from[current]
        vec1 = np.array([current[0] - prev[0], current[1] - prev[1]])
        vec2 = np.array([neighbor[0] - current[0], neighbor[1] - current[1]])
        
        if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            cos_angle = np.clip(cos_angle, -1, 1)
            turn_angle = np.arccos(cos_angle)
            smoothness_cost = turn_angle / np.pi * 10
    
    w1, w2, w3, w4 = 1.0, 1.0, 1.0, 1.0
    total_cost = w1 * distance_cost + w2 * threat_cost + w3 * terrain_cost + w4 * smoothness_cost
    
    return total_cost


def _a_star_search(grid, start, goal, threats_map=None, grid_size=100):
    """Execute A* search on the given grid."""
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
            
            if not (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]):
                continue
            
            step_cost = _calculate_multi_objective_cost(
                current, neighbor, came_from, grid, threats_map, grid_size
            )
            
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

    return None

def _create_cost_map(model, grid_size=100):
    """
    Create cost map for A* algorithm based on terrain and threats.
    
    Returns:
        terrain_map: Terrain cost map
        threats_map: Threat cost map (soft constraint)
    """
    H = model['H']
    map_range = model['map_range']
    threats = model.get('threats', np.array([]))

    scale_y = grid_size / H.shape[0]
    scale_x = grid_size / H.shape[1]
    terrain_map = zoom(H, (scale_y, scale_x), order=1)

    terrain_map = (terrain_map - np.min(terrain_map)) / (np.max(terrain_map) - np.min(terrain_map) + 1e-6) * 100

    threats_map = np.zeros((grid_size, grid_size))
    x_scale = grid_size / map_range[0]
    y_scale = grid_size / map_range[1]

    for threat in threats:
        cx, cy, cz, r = threat[0], threat[1], threat[2], threat[3]
        grid_cx = int(cx * x_scale)
        grid_cy = int(cy * y_scale)
        grid_r = int(r * x_scale)
        
        for i in range(max(0, grid_cy - grid_r * 2), min(grid_size, grid_cy + grid_r * 2)):
            for j in range(max(0, grid_cx - grid_r * 2), min(grid_size, grid_cx + grid_r * 2)):
                dist = np.sqrt((i - grid_cy)**2 + (j - grid_cx)**2)
                
                if dist <= grid_r:
                    terrain_map[i, j] = np.inf
                    threats_map[i, j] = np.inf
                elif dist <= grid_r * 1.5:
                    penalty = 100 * (1 - (dist - grid_r) / (grid_r * 0.5))
                    threats_map[i, j] = max(threats_map[i, j], penalty)

    return terrain_map, threats_map

def get_a_star_guidance_path(model, grid_size=150):
    """
    Generate multi-objective A* guidance path for A*IMOPSO.
    
    Improvements:
    1. Considers path length, threats, terrain, and smoothness
    2. Uses gradient threat penalty instead of hard constraints
    3. Considers turn angles for better smoothness
    4. Enhanced grid precision (150x150) to reduce path jaggedness
    """
    print("    -> Generating guidance path using multi-objective A* (grid: 150x150)...")
    
    terrain_map, threats_map = _create_cost_map(model, grid_size)
    map_range = model['map_range']

    start_x, start_y = model['start'][0], model['start'][1]
    end_x, end_y = model['end'][0], model['end'][1]

    start_grid = (int(start_y / map_range[1] * grid_size), int(start_x / map_range[0] * grid_size))
    goal_grid = (int(end_y / map_range[1] * grid_size), int(end_x / map_range[0] * grid_size))

    start_grid = (max(0, min(grid_size - 1, start_grid[0])), max(0, min(grid_size - 1, start_grid[1])))
    goal_grid = (max(0, min(grid_size - 1, goal_grid[0])), max(0, min(grid_size - 1, goal_grid[1])))
    
    print(f"    [DEBUG A*] Start world: ({start_x:.1f}, {start_y:.1f})")
    print(f"    [DEBUG A*] Goal world: ({end_x:.1f}, {end_y:.1f})")
    print(f"    [DEBUG A*] Start grid: {start_grid}")
    print(f"    [DEBUG A*] Goal grid: {goal_grid}")

    path_grid = _a_star_search(terrain_map, start_grid, goal_grid, threats_map, grid_size)
    
    if path_grid:
        print(f"    [DEBUG A*] Grid path first 3: {path_grid[:3]}")
        print(f"    [DEBUG A*] Grid path last 3: {path_grid[-3:]}")

    if path_grid is None:
        print("    -> Multi-objective A* failed, falling back to simple A*...")
        path_grid = _a_star_search(terrain_map, start_grid, goal_grid, None, grid_size)
        if path_grid is None:
            return None

    path_world = []
    for y_grid, x_grid in path_grid:
        x_world = (x_grid / grid_size) * map_range[0]
        y_world = (y_grid / grid_size) * map_range[1]
        
        x_world = np.clip(x_world, 1, map_range[0] - 1)
        y_world = np.clip(y_world, 1, map_range[1] - 1)
        
        path_world.append([x_world, y_world])
    
    print(f"    [DEBUG A*] Original path length: {len(path_grid)}")
    print(f"    [DEBUG A*] World coords first 3: {path_world[:3]}")
    print(f"    [DEBUG A*] World coords last 3: {path_world[-3:]}")

    if len(path_world) > model['n'] + 2:
        indices = np.linspace(0, len(path_world) - 1, model['n'] + 2, dtype=int)
        simplified_path = [path_world[i] for i in indices]
        print(f"    [DEBUG A*] Simplified path: {len(path_world)} -> {len(simplified_path)} points")
        print(f"    [DEBUG A*] Simplified first 3: {simplified_path[:3]}")
    else:
        simplified_path = path_world

    print(f"    -> Multi-objective A* succeeded ({len(simplified_path)} waypoints)")
    
    final_path = np.array(simplified_path[1:-1])
    print(f"    [DEBUG A*] Final path (without start/end): {len(final_path)} points")
    print(f"    [DEBUG A*] Final first 3: {final_path[:3]}")
    return final_path
