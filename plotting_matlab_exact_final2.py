#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高质量3D路径可视化脚本 - 论文版本
- 只生成3D视图和俯视图（不生成侧视图）
- 直接保存高分辨率矢量图
- 自动添加图例和colorbar
- 无需手动截图和拼接
"""

import numpy as np
from mayavi import mlab
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image


def _draw_cylinder(xc, yc, z_base, radius, height, color=(1, 0, 0), opacity=0.30, resolution=40):
    """绘制威胁区域圆柱体"""
    theta = np.linspace(0, 2 * np.pi, resolution)
    z = np.array([0, height])
    theta_grid, z_grid = np.meshgrid(theta, z)
    X = radius * np.cos(theta_grid) + xc
    Y = radius * np.sin(theta_grid) + yc
    Z = z_grid + z_base
    surf = mlab.mesh(X, Y, Z, color=color, opacity=opacity)
    surf.actor.property.frontface_culling = False
    surf.actor.property.backface_culling = False
    return surf


def _plot_path_line_and_midpoints(path, color_line=(0, 0, 0), tube_radius=2.5,
                                   show_control_points=True, z_offset=None,
                                   point_scale=12, color_point=None):
    """绘制路径线和控制点（线/点可分色）"""
    z = path[:, 2] + (z_offset if z_offset is not None else 0)
    mlab.plot3d(path[:, 0], path[:, 1], z, color=color_line, tube_radius=tube_radius)

    if color_point is None:
        color_point = color_line

    if show_control_points and path.shape[0] > 2:
        num_original_points = 12
        indices = np.linspace(0, len(path) - 1, num_original_points, dtype=int)
        mid = path[indices[1:-1]]
        z_mid = mid[:, 2] + (z_offset if z_offset is not None else 0)
        mlab.points3d(mid[:, 0], mid[:, 1], z_mid, color=color_point, 
                     scale_factor=point_scale, mode='sphere')


def _plot_terrain(H, map_size_x, map_size_y, colormap='summer', opacity=1.0):
    """绘制地形"""
    X, Y = np.meshgrid(np.arange(map_size_x), np.arange(map_size_y), indexing='xy')
    surf = mlab.mesh(X, Y, H, colormap=colormap, representation='surface', opacity=opacity)
    return surf


def _create_matplotlib_legend(labels, colors, save_path='legend_temp.png', ncol=1):
    """
    使用 matplotlib 创建专业图例（类似 Friedman 图的风格）
    参数:
        labels: 算法名称列表
        colors: 颜色列表 (RGB tuple, 0-1 范围)
        save_path: 图例保存路径
        ncol: 图例列数
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
    })
    fig = plt.figure(figsize=(3.5, 4.5), dpi=200)
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    patches = []
    for label, color in zip(labels, colors):
        patch = mpatches.Patch(facecolor=color, edgecolor='black', 
                               linewidth=0.8, label=label)
        patches.append(patch)
    
    legend = ax.legend(handles=patches,
                      loc='center',
                      ncol=ncol,
                      frameon=True,
                      fancybox=False,
                      edgecolor='black',
                      framealpha=1.0,
                      fontsize=14,
                      handlelength=2.5,
                      handleheight=1.4,
                      columnspacing=1.5)
    
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_linewidth(1.2)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=200, 
                transparent=True, pad_inches=0.1)
    plt.close(fig)
    
    return save_path


def _create_right_panel(labels, label_colors, terrain_min, terrain_max, 
                        colormap_name='summer', panel_height=None, save_path='right_panel.png'):
    """
    创建右侧面板：上方是算法图例，下方是地形 colorbar
    返回保存路径
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
    })
    
    fig, (ax_legend, ax_cb) = plt.subplots(2, 1, figsize=(0.5, 5.8),
                                            gridspec_kw={'height_ratios': [1.2, 5]})
    
    # 上半部分：算法图例
    ax_legend.axis('off')
    patches = []
    for label, color in zip(labels, label_colors):
        patch = mpatches.Patch(facecolor=color, edgecolor='black', 
                               linewidth=0.8, label=label)
        patches.append(patch)
    legend = ax_legend.legend(handles=patches, loc='center', frameon=True,
                              fancybox=False, edgecolor='black', framealpha=1.0,
                              fontsize=13, handlelength=2.2, handleheight=1.2)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_linewidth(1.0)
    
    # 下半部分：地形 colorbar
    cmap = cm.get_cmap(colormap_name)
    norm = mcolors.Normalize(vmin=terrain_min, vmax=terrain_max)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cb, orientation='vertical')
    tick_step = 50
    ticks = np.arange(terrain_min, terrain_max + 1, tick_step)
    cb.set_ticks(ticks)
    cb.ax.tick_params(labelsize=18)
    
    fig.subplots_adjust(hspace=0.2)
    plt.savefig(save_path, bbox_inches='tight', dpi=200, transparent=False, pad_inches=0.02)
    plt.close(fig)
    return save_path


def _overlay_legend_on_image(base_image_path, legend_path, output_path, 
                            position='top-right', margin=(50, 50)):
    """
    将图例叠加到图片上
    参数:
        base_image_path: 基础图片路径
        legend_path: 图例图片路径
        output_path: 输出图片路径
        position: 图例位置 ('top-right', 'top-left', 'bottom-right', 'bottom-left')
        margin: (x_margin, y_margin) 边距
    """
    
    # 打开图片
    img_base = Image.open(base_image_path).convert('RGBA')
    img_legend = Image.open(legend_path).convert('RGBA')
    
    # 计算图例位置
    legend_w, legend_h = img_legend.size
    base_w, base_h = img_base.size
    
    if position == 'top-right':
        x = base_w - legend_w - margin[0]
        y = margin[1]
    elif position == 'top-left':
        x = margin[0]
        y = margin[1]
    elif position == 'bottom-right':
        x = base_w - legend_w - margin[0]
        y = base_h - legend_h - margin[1]
    elif position == 'bottom-left':
        x = margin[0]
        y = base_h - legend_h - margin[1]
    else:
        x, y = margin
    
    # 叠加图例
    img_base.paste(img_legend, (x, y), img_legend)
    
    # 转换回 RGB 并保存
    img_result = img_base.convert('RGB')
    img_result.save(output_path, quality=95)
    
    return output_path


def _crop_white_borders(image_path, output_path=None, border_threshold=250, padding=5):
    """
    裁剪图片周围的白色边框，保留 padding 像素安全边距
    """
    from PIL import ImageChops
    
    img = Image.open(image_path).convert('RGB')
    
    bg = Image.new('RGB', img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg)
    diff = diff.convert('L')
    bbox = diff.getbbox()
    
    if bbox:
        x0, y0, x1, y1 = bbox
        x0 = max(0, x0 - padding)
        y0 = max(0, y0 - padding)
        x1 = min(img.width, x1 + padding)
        y1 = min(img.height, y1 + padding)
        img_cropped = img.crop((x0, y0, x1, y1))
        
        # 保存
        if output_path is None:
            output_path = image_path
        img_cropped.save(output_path, quality=95)
        
        return output_path
    else:
        # 如果没有找到边界，返回原图
        return image_path


def _combine_images_horizontal(left_image_path, right_image_path, output_path, gap=20, valign='center'):
    """
    将两张图片左右拼接成一张图片
    参数:
        left_image_path: 左侧图片路径
        right_image_path: 右侧图片路径
        output_path: 输出图片路径
        gap: 两张图片之间的间隙（像素）
        valign: 垂直对齐方式 ('center', 'top', 'bottom')
    
    支持的输出格式：
        - .png: 使用 PIL 保存（高质量）
        - .eps: 使用 matplotlib 保存（矢量格式）
    """
    img_left = Image.open(left_image_path).convert('RGB')
    img_right = Image.open(right_image_path).convert('RGB')
    
    left_w, left_h = img_left.size
    right_w, right_h = img_right.size
    
    total_width = left_w + gap + right_w
    total_height = max(left_h, right_h)
    
    combined_img = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    
    def _calc_y(img_h):
        if valign == 'top':
            return 0
        elif valign == 'bottom':
            return total_height - img_h
        else:
            return (total_height - img_h) // 2
    
    combined_img.paste(img_left, (0, _calc_y(left_h)))
    combined_img.paste(img_right, (left_w + gap, _calc_y(right_h)))
    
    # 根据输出格式选择保存方法
    output_ext = os.path.splitext(output_path)[1].lower()
    
    if output_ext == '.eps':
        # 使用 matplotlib 保存为 EPS 格式（矢量格式）
        fig = plt.figure(figsize=(total_width / 100, total_height / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.imshow(combined_img)
        plt.savefig(output_path, format='eps', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"  ✅ EPS 格式拼接图已保存")
    else:
        # 默认使用 PIL 保存（PNG 等格式）
        combined_img.save(output_path, quality=95)
    
    return output_path


def _draw_custom_axes_and_grid(map_size_x, map_size_y, scene_z_max,
                               draw_x=True, draw_y=True, draw_z=True,
                               yz_grid_at_far_side=False,
                               tick_scale_override=None, label_scale_override=None):
    """绘制自定义坐标轴和网格"""
    grid_color = (0.7, 0.7, 0.7)
    axis_color = (0, 0, 0)
    label_color = (0, 0, 0)
    tick_length = 30
    tick_interval = 200
    tick_font_scale = tick_scale_override if tick_scale_override else 26
    label_font_scale = label_scale_override if label_scale_override else 30

    # XY 平面网格
    if draw_x and draw_y:
        for x_line in np.arange(0, map_size_x + 1, tick_interval):
            mlab.plot3d([x_line, x_line], [0, map_size_y], [0, 0], 
                       color=grid_color, tube_radius=0.5)
        for y_line in np.arange(0, map_size_y + 1, tick_interval):
            mlab.plot3d([0, map_size_x], [y_line, y_line], [0, 0], 
                       color=grid_color, tube_radius=0.5)
    
    # YZ 平面网格
    if draw_y and draw_z:
        grid_x_pos = map_size_x if yz_grid_at_far_side else 0
        for y_line in np.arange(0, map_size_y + 1, tick_interval):
            mlab.plot3d([grid_x_pos, grid_x_pos], [y_line, y_line], 
                       [0, scene_z_max], color=grid_color, tube_radius=0.5)
        for z_line in np.arange(0, scene_z_max + 1, 100):
            mlab.plot3d([grid_x_pos, grid_x_pos], [0, map_size_y], 
                       [z_line, z_line], color=grid_color, tube_radius=0.5)
    
    # XZ 平面网格
    if draw_x and draw_z:
        for x_line in np.arange(0, map_size_x + 1, tick_interval):
            mlab.plot3d([x_line, x_line], [map_size_y, map_size_y], 
                       [0, scene_z_max], color=grid_color, tube_radius=0.5)
        for z_line in np.arange(0, scene_z_max + 1, 100):
            mlab.plot3d([0, map_size_x], [map_size_y, map_size_y], 
                       [z_line, z_line], color=grid_color, tube_radius=0.5)

    # X 轴
    if draw_x:
        mlab.plot3d([0, map_size_x], [0, 0], [0, 0], color=axis_color, tube_radius=1.5)
        mlab.text3d(map_size_x / 2, -tick_length * 5, 0, 'x [m]', 
                   scale=label_font_scale, color=label_color)
        for x in np.arange(0, map_size_x + 1, tick_interval):
            mlab.plot3d([x, x], [0, -tick_length], [0, 0], 
                       color=axis_color, tube_radius=1)
            mlab.text3d(x, -tick_length * 3, 0, str(x), 
                       scale=tick_font_scale, color=label_color)
    
    # Y 轴
    if draw_y:
        mlab.plot3d([0, 0], [0, map_size_y], [0, 0], color=axis_color, tube_radius=1.5)
        mlab.text3d(-tick_length * 8, map_size_y / 2, 0, 'y [m]', 
                   scale=label_font_scale, color=label_color)
        for y in np.arange(0, map_size_y + 1, tick_interval):
            mlab.plot3d([-tick_length, 0], [y, y], [0, 0], 
                       color=axis_color, tube_radius=1)
            mlab.text3d(-tick_length * 4, y, 0, str(y), 
                       scale=tick_font_scale, color=label_color)
    
    # Z 轴
    if draw_z:
        z_axis_x_pos = map_size_x if yz_grid_at_far_side else 0
        mlab.plot3d([z_axis_x_pos, z_axis_x_pos], [0, 0], [0, scene_z_max], 
                   color=axis_color, tube_radius=1.5)
        mlab.text3d(z_axis_x_pos, -tick_length * 8, scene_z_max / 2, 'z [m]', 
                   scale=label_font_scale, color=label_color)
        for z in np.arange(0, scene_z_max + 1, 100):
            mlab.plot3d([z_axis_x_pos - tick_length, z_axis_x_pos], [0, 0], 
                       [z, z], color=axis_color, tube_radius=1)
            if z > 0:
                mlab.text3d(z_axis_x_pos, -tick_length * 3.5, z, str(int(z)), 
                           scale=tick_font_scale, color=label_color)


def plot_and_save_paper_figures(paths_absolute, path_labels, model, 
                                save_dir="paper_figures", 
                                scene_name="scene_1",
                                experiment_group=None,
                                show_control_points=True, 
                                raise_topview=True,
                                dpi=300):
    """
    生成并保存论文用的高质量3D可视化图
    
    参数:
        paths_absolute: 路径列表
        path_labels: 算法名称列表
        experiment_group: 实验组编号（可选，用于区分不同实验组，会创建子文件夹）
        model: 环境模型字典
        save_dir: 保存目录
        scene_name: 场景名称
        show_control_points: 是否显示控制点
        raise_topview: 俯视图中路径是否抬高
        dpi: 图片分辨率
    """
    H = model['H']
    threats = model.get('threats', np.zeros((0, 4)))
    map_size_x, map_size_y = model['map_range']

    scene_z_max = 400
    cyl_height = 400

    # Science 经典学术配色（沉稳、高区分度、色盲友好）
    colors = [
        (0.882, 0.506, 0.510),  # #E18182 粉红 - A*IMOPSO（主角）
        (0.976, 0.808, 0.612),  # #F9CE9C 浅橙
        (0.165, 0.298, 0.443),  # #2A4C71 深海蓝
        (0.784, 0.722, 0.831),  # #C8B8D4 淡紫
        (0.000, 0.627, 0.529),  # #00A087 墨绿色
        (0.169, 0.169, 0.169),  # #2B2B2B 深灰
        (0.522, 0.569, 0.706),  # #8491B4 薰衣草灰
        (0.612, 0.122, 0.388),  # #9C1F63 酒红
    ]

    if not paths_absolute:
        print("警告: 没有可绘制的路径。")
        return

    # 创建保存目录（根据实验组创建子文件夹）
    if experiment_group is not None:
        # 根据实验组编号创建对应的子文件夹
        if experiment_group == 1:
            save_dir = os.path.join(save_dir, "pso_variants")
        elif experiment_group == 2:
            save_dir = os.path.join(save_dir, "classic_algorithms")
        else:
            # 如果是其他数字，直接使用数字作为文件夹名
            save_dir = os.path.join(save_dir, str(experiment_group))
    os.makedirs(save_dir, exist_ok=True)

    # 计算地形高度范围
    terrain_max = np.ceil(H.max() / 50) * 50
    
    # 重新排序算法，将 A*IMOPSO 放在第一个
    # 查找 A*IMOPSO 的索引
    aimopso_idx = None
    for i, label in enumerate(path_labels):
        if 'A*IMOPSO' in label or 'A*MOPSO' in label or 'AIMOPSO' in label:
            aimopso_idx = i
            break
    
    # 如果找到 A*IMOPSO，重新排序（只排 labels 和 paths，不动 colors）
    if aimopso_idx is not None and aimopso_idx != 0:
        path_labels_reordered = [path_labels[aimopso_idx]] + path_labels[:aimopso_idx] + path_labels[aimopso_idx+1:]
        paths_absolute_reordered = [paths_absolute[aimopso_idx]] + paths_absolute[:aimopso_idx] + paths_absolute[aimopso_idx+1:]
        
        path_labels = path_labels_reordered
        paths_absolute = paths_absolute_reordered

    # ==================== 1. 3D 视图 ====================
    print(f"生成 3D 视图...")
    fig_3d = mlab.figure('3D View', bgcolor=(1, 1, 1), size=(1200, 800))
    
    # 绘制地形
    terrain_surface = _plot_terrain(H, map_size_x, map_size_y, 
                                    colormap='summer', opacity=1.0)
    terrain_surface.module_manager.scalar_lut_manager.data_range = [50, terrain_max]
    
    # 3D视图不放colorbar，只在俯视图中显示（节省空间）
    num_labels = int((terrain_max - 50) / 50) + 1
    
    # 绘制坐标轴和网格
    _draw_custom_axes_and_grid(map_size_x, map_size_y, scene_z_max,
                               draw_x=True, draw_y=True, draw_z=True,
                               yz_grid_at_far_side=True)
    
    # 绘制威胁区域
    for (x0, y0, z0, R) in threats:
        _draw_cylinder(x0, y0, z0, R, cyl_height, color=(1, 0, 0), opacity=0.30)
    
    # 绘制路径（主角A*IMOPSO加粗，航路点统一金黄色）
    waypoint_color = (1, 0.84, 0)  # 金黄色航路点
    for i, path in enumerate(paths_absolute):
        _plot_path_line_and_midpoints(path, color_line=colors[i % len(colors)], 
                                      tube_radius=5.0, point_scale=14,
                                      color_point=waypoint_color,
                                      show_control_points=show_control_points)
    
    # 绘制起点和终点
    start_point_abs = paths_absolute[0][0, :]
    end_point_abs = paths_absolute[0][-1, :]
    mlab.points3d(start_point_abs[0], start_point_abs[1], start_point_abs[2], 
                 color=(0, 0, 0), scale_factor=18, mode='cube')
    mlab.points3d(end_point_abs[0], end_point_abs[1], end_point_abs[2], 
                 color=(0, 0, 0), scale_factor=18, mode='sphere')
    
    # 设置视角
    mlab.view(azimuth=-135, elevation=65, distance='auto')
    
    mlab.text(0.5, 0.06, '(a) 3D path view', width=0.55, color=(0, 0, 0))
    
    # 保存3D图片（不叠加图例，图例只在俯视图显示）
    save_path_3d = os.path.join(save_dir, f"{scene_name}_3d_view.png")
    mlab.savefig(save_path_3d, size=(900, 600), magnification=dpi/100)
    
    # 裁剪白色边框
    _crop_white_borders(save_path_3d, save_path_3d)
    
    print(f"✅ 3D 视图已保存: {save_path_3d}")
    
    mlab.close(fig_3d)

    # ==================== 2. 俯视图 ====================
    print(f"生成俯视图...")
    fig_top = mlab.figure('Top View', bgcolor=(1, 1, 1), size=(1200, 900))
    
    # 绘制地形
    surf_top = _plot_terrain(H, map_size_x, map_size_y, 
                            colormap='summer', opacity=1.0)
    surf_top.module_manager.scalar_lut_manager.data_range = [50, terrain_max]
    surf_top.actor.actor.force_opaque = True
    
    # colorbar 不在 Mayavi 中渲染，改为后期用 matplotlib 画在右侧面板
    
    # 绘制威胁区域（圆圈）
    z_circle = np.max(H) + 2
    theta = np.linspace(0, 2 * np.pi, 800)
    for (x0, y0, _, R) in threats:
        for r_cur in [R, R - 20, R - 40]:
            if r_cur > 0:
                mlab.plot3d(x0 + r_cur * np.cos(theta), 
                           y0 + r_cur * np.sin(theta),
                           np.full_like(theta, z_circle), 
                           color=(1, 0, 0), tube_radius=1.2)
        mlab.points3d(x0, y0, z_circle, color=(1, 0, 0), 
                     scale_factor=10, mode='sphere')
    
    # 绘制路径（主角A*IMOPSO加粗，航路点统一金黄色）
    waypoint_color_top = (1, 0.84, 0)
    z_offset = 3 if raise_topview else 0
    for i, path in enumerate(paths_absolute):
        _plot_path_line_and_midpoints(path, color_line=colors[i % len(colors)], 
                                      tube_radius=7.0, point_scale=20,
                                      color_point=waypoint_color_top,
                                      show_control_points=show_control_points,
                                      z_offset=z_offset)
    
    # 绘制起点和终点
    mlab.points3d(start_point_abs[0], start_point_abs[1], z_circle + z_offset, 
                 color=(0, 0, 0), scale_factor=18, mode='cube')
    mlab.points3d(end_point_abs[0], end_point_abs[1], z_circle + z_offset, 
                 color=(0, 0, 0), scale_factor=18, mode='sphere')
    
    # 绘制坐标轴（parallel projection 下 3D 文字会缩小，需要补偿放大）
    _draw_custom_axes_and_grid(map_size_x, map_size_y, 0, 
                               draw_x=True, draw_y=True, draw_z=False,
                               tick_scale_override=36, label_scale_override=42)
    
    # 设置视角（模拟 View along +Z axis + Toggle parallel projection）
    # 俯视图：elevation=0（从上往下看XY平面）
    mlab.view(azimuth=0, elevation=0, distance='auto')
    
    # Toggle parallel projection: 启用正交投影
    fig_top.scene.camera.parallel_projection = True
    
    # 调整正交投影的缩放比例（parallel_scale越大，内容越小）
    # 减小parallel_scale以确保y轴刻度不被截断
    fig_top.scene.camera.parallel_scale = 950
    
    mlab.text(0.5, 0.06, '(b) Top view of the path', width=0.75, color=(0, 0, 0))
    
    # 保存纯净的俯视图（无 colorbar，无图例）
    temp_path_top = os.path.join(save_dir, f"{scene_name}_top_view_temp.png")
    mlab.savefig(temp_path_top, size=(1100, 600), magnification=dpi/100)
    
    # 裁剪白色边框
    _crop_white_borders(temp_path_top, temp_path_top)
    
    mlab.close(fig_top)
    
    # 用 matplotlib 创建右侧面板（图例 + colorbar 上下排列）
    panel_path = os.path.join(save_dir, 'right_panel_temp.png')
    _create_right_panel(path_labels, colors, 50, terrain_max,
                        colormap_name='summer', save_path=panel_path)
    
    # 并排拼接：俯视图 + 右侧面板（顶部对齐，无间隙）
    save_path_top = os.path.join(save_dir, f"{scene_name}_top_view.png")
    _combine_images_horizontal(temp_path_top, panel_path, save_path_top, gap=-160, valign='top')
    
    # 删除临时文件
    os.remove(temp_path_top)
    os.remove(panel_path)
    
    print(f"✅ 俯视图已保存: {save_path_top}")

    # ==================== 3. 拼接两张图片 ====================
    print(f"拼接3D视图和俯视图...")
    # 保存 PNG 格式
    combined_path_png = os.path.join(save_dir, f"{scene_name}_combined.png")
    _combine_images_horizontal(save_path_3d, save_path_top, combined_path_png, gap=3)
    print(f"✅ PNG 拼接图已保存: {combined_path_png}")
    
    # 保存 EPS 格式
    combined_path_eps = os.path.join(save_dir, f"{scene_name}_combined.eps")
    _combine_images_horizontal(save_path_3d, save_path_top, combined_path_eps, gap=3)
    print(f"✅ EPS 拼接图已保存: {combined_path_eps}")

    print(f"\n✅ 所有图片已保存到: {save_dir}")
    print(f"   - {scene_name}_3d_view.png (3D视图)")
    print(f"   - {scene_name}_top_view.png (俯视图)")
    print(f"   - {scene_name}_combined.png (PNG拼接图)")
    print(f"   - {scene_name}_combined.eps (EPS拼接图) ⭐")


# 示例使用
if __name__ == "__main__":
    print("这是一个用于生成论文高质量3D可视化图的模块。")
    print("请从 compare_algorithms_visual_cached.py 中调用。")
 