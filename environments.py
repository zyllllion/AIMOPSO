
import numpy as np


def _terrain_scene1(x, y):
    """Terrain for Scene 1 and Scene 3"""
    def Z1(x, y, a=50, b=60):
        return a * np.sin(x / b) + np.cos(y / b)

    def Z2(x, y, h_list, x_list, y_list, a_list, b_list):
        z = np.zeros_like(x)
        for i in range(len(h_list)):
            z += h_list[i] * np.exp(-(((x - x_list[i]) ** 2) / (2 * a_list[i] ** 2) +
                                      ((y - y_list[i]) ** 2) / (2 * b_list[i] ** 2)))
        return z

    h_list = [120, 200, 150, 180]
    x_list = [200, 500, 700, 300]
    y_list = [300, 600, 200, 800]
    a_list = [60, 80, 70, 90]
    b_list = [80, 100, 60, 70]
    z = Z1(x, y) + Z2(x, y, h_list, x_list, y_list, a_list, b_list)
    return z - z.min() if z.min() < 0 else z


def _terrain_scene2(x, y):
    """Complex terrain for Scene 2"""
    def Z1(x, y, a1=60, b1=70, a2=40, b2=50):
        return a1 * np.sin(x / b1) + np.cos(y / b1) + a2 * np.sin(y / b2) * np.cos(x / b2)

    def Z2(x, y, h_list, x_list, y_list, a_list, b_list):
        z = np.zeros_like(x)
        for i in range(len(h_list)):
            z += h_list[i] * np.exp(-(((x - x_list[i]) ** 2) / (2 * a_list[i] ** 2) +
                                      ((y - y_list[i]) ** 2) / (2 * b_list[i] ** 2)))
        return z

    h_list = [80, 220, 180, 80, 160, 190]
    x_list = [150, 400, 650, 850, 300, 700]
    y_list = [200, 500, 250, 700, 600, 800]
    a_list = [50, 90, 70, 80, 60, 100]
    b_list = [60, 100, 50, 70, 80, 90]
    z = Z1(x, y) + Z2(x, y, h_list, x_list, y_list, a_list, b_list)
    return z - z.min() if z.min() < 0 else z


def _threats_scene1():
    return np.array([
        [250, 250, 0, 80], [700, 500, 0, 70],
        [400, 700, 0, 60], [600, 150, 0, 90]
    ])

def _threats_scene2():
    return np.array([
        [650, 300, 0, 70], [700, 500, 0, 50], [300, 600, 0, 90],
        [700, 850, 0, 70], [500, 400, 0, 60],
    ])

def _threats_scene3():
    return np.array([
        [300, 300, 0, 60], [500, 500, 0, 70], [700, 200, 0, 80],
        [200, 700, 0, 90], [400, 800, 0, 50], [800, 500, 0, 75],
    ])

def _threats_scene4():
    """Threat zones for Scene 4"""
    return np.array([
        [200, 500, 0, 80],
        [400, 300, 0, 60],
        [500, 700, 0, 100],
        [750, 450, 0, 75],
        [800, 200, 0, 50],
        [150, 800, 0, 65]
    ])


def create_scene_model(scene_id):
    MAPSIZE_X, MAPSIZE_Y = 1000, 1000
    x = np.linspace(0, MAPSIZE_X, MAPSIZE_X)
    y = np.linspace(0, MAPSIZE_Y, MAPSIZE_Y)
    X, Y = np.meshgrid(x, y, indexing='xy')

    common_params = {
        'n': 10, 'map_range': [MAPSIZE_X, MAPSIZE_Y],
        'xmin': 1, 'xmax': MAPSIZE_X, 'ymin': 1, 'ymax': MAPSIZE_Y,
    }

    RELATIVE_Z_MIN = 50
    RELATIVE_Z_MAX = 150

    if scene_id == 1:
        model = {
            'name': 'Scene 1', 
            'start': np.array([50, 50, 150]),
            'end': np.array([900, 900, 200]),
            'H': _terrain_scene1(X, Y),
            'zmin': RELATIVE_Z_MIN, 'zmax': RELATIVE_Z_MAX,
            'threats': _threats_scene1(), **common_params
        }
    elif scene_id == 2:
        model = {
            'name': 'Scene 2', 
            'start': np.array([80, 80, 80]),
            'end': np.array([850, 850, 160]),
            'H': _terrain_scene2(X, Y),
            'zmin': RELATIVE_Z_MIN, 'zmax': RELATIVE_Z_MAX,
            'threats': _threats_scene2(), **common_params
        }
    elif scene_id == 3:
        model = {
            'name': 'Scene 3', 
            'start': np.array([50, 50, 80]),
            'end': np.array([800, 800, 130]),
            'H': _terrain_scene1(X, Y),
            'zmin': RELATIVE_Z_MIN, 'zmax': RELATIVE_Z_MAX,
            'threats': _threats_scene3(), **common_params
        }
    elif scene_id == 4:
        model = {
            'name': 'Scene 4', 
            'start': np.array([50, 50, 100]),
            'end': np.array([800, 800, 140]),
            'H': _terrain_scene2(X, Y),
            'zmin': RELATIVE_Z_MIN, 'zmax': RELATIVE_Z_MAX,
            'threats': _threats_scene4(), **common_params
        }
    else:
        raise ValueError(f"Unknown scene ID: {scene_id}")

    return model
def create_scene1_model(): return create_scene_model(1)
def create_scene2_model(): return create_scene_model(2)
def create_scene3_model(): return create_scene_model(3)
def create_scene4_model(): return create_scene_model(4)
