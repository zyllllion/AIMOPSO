# A*IMOPSO Paper Code

精简的论文复现代码，用于 "A*-Guided Heuristic Multi-Objective PSO for UAV 3D Path Planning"。

## 文件说明

### 核心算法
| 文件 | 说明 |
|------|------|
| `coordinates.py` | 球坐标与笛卡尔坐标转换 |
| `a_star_guidance.py` | A* 引导路径生成 |
| `cost_function.py` | 多目标代价函数 (J1-J4) |
| `pso_operators.py` | PSO 基础算子（支配关系） |
| `aimopso_operators.py` | A*IMOPSO 特有算子（非支配排序、锦标赛选择、多项式变异） |
| `aimopso_runner.py` | A*IMOPSO 主算法 |
| `environments.py` | 4 个场景的地形与威胁模型 |

### 实验脚本
| 文件 | 说明 |
|------|------|
| `run_full_mo_benchmark2.py` | 主对比实验（A*IMOPSO vs NSGA-II, SPEA2, SMS-EMOA, NSGA-III, RVEA, AGEMOEA2） |
| `run_ablation_study.py` | 消融实验（A*IMOPSO / A_MOPSO / IMOPSO / MOPSO） |

### 结果分析
| 文件 | 说明 |
|------|------|
| `analyze_all_scenes_unified.py` | 计算 HV、IGD，生成 Friedman 排名图及论文表格 |
| `analyze_ablation_study.py` | 消融实验统计分析 |

### 可视化（可选）
| 文件 | 说明 |
|------|------|
| `plotting_matlab_exact_final2.py` | 3D 路径与俯视图绘制 |
| `visualize_aimopso_diverse_solutions.py` | A*IMOPSO 多样化解集可视化 |
| `algorithm_cache_manager.py` | 可视化脚本的缓存管理 |

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行对比实验（示例：场景2，A*IMOPSO，30次独立运行）
python run_full_mo_benchmark2.py --scene 2 --algorithm A*IMOPSO

# 3. 运行消融实验（示例）
python run_ablation_study.py --algorithm A_IMOPSO --scene 1 --start_run 1 --end_run 30

# 4. 分析结果（需先完成步骤2，结果在 run_results_mo_sota/ 下）
python analyze_all_scenes_unified.py
```

## 输出目录

- `run_results_mo_sota/`：对比实验结果（CSV）
- `run_results_ablation/`：消融实验结果

## 依赖说明

- **必需**：numpy, scipy, pandas, pymoo, matplotlib  
- **3D 可视化**：mayavi, pillow（仅 `visualize_aimopso_diverse_solutions.py` 需要）
