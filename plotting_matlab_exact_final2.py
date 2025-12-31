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


def _plot_path_line_and_midpoints(path, color_line=(0, 0, 0), tube_radius=2.0, 
                                   show_control_points=True, z_offset=None):
    """绘制路径线和控制点"""
    z = path[:, 2] + (z_offset if z_offset is not None else 0)
    mlab.plot3d(path[:, 0], path[:, 1], z, color=color_line, tube_radius=tube_radius)

    if show_control_points and path.shape[0] > 2:
        num_original_points = 12
        indices = np.linspace(0, len(path) - 1, num_original_points, dtype=int)
        mid = path[indices[1:-1]]
        z_mid = mid[:, 2] + (z_offset if z_offset is not None else 0)
        mlab.points3d(mid[:, 0], mid[:, 1], z_mid, color=color_line, 
                     scale_factor=8, mode='sphere')


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
    # 创建图例 figure（调整为竖向尺寸）
    fig = plt.figure(figsize=(3, 4), dpi=150)  # 竖向布局
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # 创建图例 patches
    patches = []
    for label, color in zip(labels, colors):
        # 确保颜色格式正确（matplotlib 需要 RGB tuple）
        patch = mpatches.Patch(facecolor=color, edgecolor='black', 
                               linewidth=0.8, label=label)
        patches.append(patch)
    
    # 创建图例（放大字体和间距）
    legend = ax.legend(handles=patches,
                      loc='center',
                      ncol=ncol,
                      frameon=True,
                      fancybox=False,
                      edgecolor='black',
                      framealpha=1.0,
                      fontsize=12,  # 放大字体
                      handlelength=2.0,  # 放大颜色块
                      handleheight=1.2,
                      columnspacing=1.5)
    
    # 设置图例背景为白色
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_linewidth(1.0)
    
    # 保存图例（透明背景）
    plt.savefig(save_path, bbox_inches='tight', dpi=150, 
                transparent=True, pad_inches=0.1)
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


def _crop_white_borders(image_path, output_path=None, border_threshold=250):
    """
    裁剪图片周围的白色边框
    参数:
        image_path: 输入图片路径
        output_path: 输出图片路径（如果为None，则覆盖原图）
        border_threshold: 白色阈值（0-255，接近255的像素被视为白色）
    """
    from PIL import ImageChops
    
    img = Image.open(image_path).convert('RGB')
    
    # 创建白色背景
    bg = Image.new('RGB', img.size, (255, 255, 255))
    
    # 计算差异
    diff = ImageChops.difference(img, bg)
    
    # 转换为灰度并获取边界框
    diff = diff.convert('L')
    bbox = diff.getbbox()
    
    if bbox:
        # 裁剪图片
        img_cropped = img.crop(bbox)
        
        # 保存
        if output_path is None:
            output_path = image_path
        img_cropped.save(output_path, quality=95)
        
        return output_path
    else:
        # 如果没有找到边界，返回原图
        return image_path


def _combine_images_horizontal(left_image_path, right_image_path, output_path, gap=20):
    """
    将两张图片左右拼接成一张图片
    参数:
        left_image_path: 左侧图片路径
        right_image_path: 右侧图片路径
        output_path: 输出图片路径
        gap: 两张图片之间的间隙（像素）
    
    支持的输出格式：
        - .png: 使用 PIL 保存（高质量）
        - .eps: 使用 matplotlib 保存（矢量格式）
    """
    # 打开图片
    img_left = Image.open(left_image_path).convert('RGB')
    img_right = Image.open(right_image_path).convert('RGB')
    
    # 获取图片尺寸
    left_w, left_h = img_left.size
    right_w, right_h = img_right.size
    
    # 计算拼接后的图片尺寸（高度取最大值）
    total_width = left_w + gap + right_w
    total_height = max(left_h, right_h)
    
    # 创建新图片（白色背景）
    combined_img = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    
    # 粘贴左侧图片（垂直居中）
    left_y = (total_height - left_h) // 2
    combined_img.paste(img_left, (0, left_y))
    
    # 粘贴右侧图片（垂直居中）
    right_y = (total_height - right_h) // 2
    combined_img.paste(img_right, (left_w + gap, right_y))
    
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
                               yz_grid_at_far_side=False):
    """绘制自定义坐标轴和网格"""
    grid_color = (0.7, 0.7, 0.7)
    axis_color = (0, 0, 0)
    label_color = (0, 0, 0)
    tick_length = 20
    tick_interval = 200

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
        mlab.text3d(map_size_x / 2, -tick_length * 4, 0, 'x [m]', 
                   scale=15, color=label_color)
        for x in np.arange(0, map_size_x + 1, tick_interval):
            mlab.plot3d([x, x], [0, -tick_length], [0, 0], 
                       color=axis_color, tube_radius=1)
            mlab.text3d(x, -tick_length * 2.5, 0, str(x), 
                       scale=12, color=label_color)
    
    # Y 轴
    if draw_y:
        mlab.plot3d([0, 0], [0, map_size_y], [0, 0], color=axis_color, tube_radius=1.5)
        mlab.text3d(-tick_length * 7, map_size_y / 2, 0, 'y [m]', 
                   scale=15, color=label_color)
        for y in np.arange(0, map_size_y + 1, tick_interval):
            mlab.plot3d([-tick_length, 0], [y, y], [0, 0], 
                       color=axis_color, tube_radius=1)
            mlab.text3d(-tick_length * 3.5, y, 0, str(y), 
                       scale=12, color=label_color)
    
    # Z 轴
    if draw_z:
        z_axis_x_pos = map_size_x if yz_grid_at_far_side else 0
        mlab.plot3d([z_axis_x_pos, z_axis_x_pos], [0, 0], [0, scene_z_max], 
                   color=axis_color, tube_radius=1.5)
        mlab.text3d(z_axis_x_pos, -tick_length * 7, scene_z_max / 2, 'z [m]', 
                   scale=15, color=label_color)
        for z in np.arange(0, scene_z_max + 1, 100):
            mlab.plot3d([z_axis_x_pos - tick_length, z_axis_x_pos], [0, 0], 
                       [z, z], color=axis_color, tube_radius=1)
            if z > 0:
                mlab.text3d(z_axis_x_pos, -tick_length * 3, z, str(int(z)), 
                           scale=12, color=label_color)


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

    # 定义颜色（与论文中常用的颜色一致）
    colors = [
        (1, 0, 1),      # 品红 - A*IMOPSO专用
        (0, 0, 1),      # 蓝色
        (1, 0, 0),      # 红色
        (1, 1, 0),      # 黄色
        (0, 1, 1),      # 青色
        (0.5, 0, 0.5),  # 紫色
        (1, 0.5, 0),    # 橙色
        (0, 0, 0)       # 黑色
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
    
    # 如果找到 A*IMOPSO，重新排序
    if aimopso_idx is not None and aimopso_idx != 0:
        # 重新排序 labels 和 colors
        path_labels_reordered = [path_labels[aimopso_idx]] + path_labels[:aimopso_idx] + path_labels[aimopso_idx+1:]
        colors_reordered = [colors[aimopso_idx]] + colors[:aimopso_idx] + colors[aimopso_idx+1:]
        paths_absolute_reordered = [paths_absolute[aimopso_idx]] + paths_absolute[:aimopso_idx] + paths_absolute[aimopso_idx+1:]
        
        # 更新变量
        path_labels = path_labels_reordered
        colors = colors_reordered
        paths_absolute = paths_absolute_reordered

    # ==================== 1. 3D 视图 ====================
    print(f"生成 3D 视图...")
    fig_3d = mlab.figure('3D View', bgcolor=(1, 1, 1), size=(1200, 900))
    
    # 绘制地形
    terrain_surface = _plot_terrain(H, map_size_x, map_size_y, 
                                    colormap='summer', opacity=1.0)
    terrain_surface.module_manager.scalar_lut_manager.data_range = [50, terrain_max]
    
    # 添加 colorbar（右侧）
    colorbar = mlab.colorbar(terrain_surface, title='', orientation='vertical', nb_labels=None)
    # 设置 colorbar 位置（向左移动，靠近路径规划图）
    colorbar.scalar_bar_representation.position = [0.86, 0.25]  # [x, y] - 向左移到 0.88
    colorbar.scalar_bar_representation.position2 = [0.04, 0.35]  # [width, height] - 缩小一半
    # 设置标签数量和格式
    num_labels = int((terrain_max - 50) / 50) + 1
    colorbar.scalar_bar.number_of_labels = num_labels
    colorbar.scalar_bar.label_format = '%.0f'
    # 设置标签样式（减小字体，与坐标轴数字一致）
    label_props = colorbar.label_text_property
    label_props.color = (0, 0, 0)
    label_props.font_size = 10  # 减小字体
    label_props.bold = False
    label_props.italic = False
    label_props.font_family = 'arial'
    # 设置 colorbar 标题样式
    title_props = colorbar.title_text_property
    title_props.font_size = 10
    title_props.bold = False
    title_props.italic = False
    title_props.font_family = 'arial'
    
    # 绘制坐标轴和网格
    _draw_custom_axes_and_grid(map_size_x, map_size_y, scene_z_max,
                               draw_x=True, draw_y=True, draw_z=True,
                               yz_grid_at_far_side=True)
    
    # 绘制威胁区域
    for (x0, y0, z0, R) in threats:
        _draw_cylinder(x0, y0, z0, R, cyl_height, opacity=0.30)
    
    # 绘制路径
    for i, path in enumerate(paths_absolute):
        _plot_path_line_and_midpoints(path, color_line=colors[i % len(colors)], 
                                      show_control_points=show_control_points)
    
    # 绘制起点和终点
    start_point_abs = paths_absolute[0][0, :]
    end_point_abs = paths_absolute[0][-1, :]
    mlab.points3d(start_point_abs[0], start_point_abs[1], start_point_abs[2], 
                 color=(0, 0, 0), scale_factor=14, mode='cube')
    mlab.points3d(end_point_abs[0], end_point_abs[1], end_point_abs[2], 
                 color=(0, 0, 0), scale_factor=14, mode='sphere')
    
    # 设置视角
    mlab.view(azimuth=-135, elevation=65, distance='auto')
    
    # 在Mayavi场景中添加标题（使用text方法精确定位）
    mlab.text(0.5, 0.1, '(a) 3D path view', width=0.3, color=(0, 0, 0))
    
    # 保存图片（包含标题）
    temp_path_3d = os.path.join(save_dir, f"{scene_name}_3d_view_temp.png")
    mlab.savefig(temp_path_3d, size=(800, 500), magnification=dpi/100)
    

    # 创建 matplotlib 图例（单列竖向排列）
    legend_path = os.path.join(save_dir, 'legend_temp.png')
    _create_matplotlib_legend(path_labels, colors, legend_path, ncol=1)  # ncol=1 竖着放
    
    # 叠加图例到 3D 图（向左移动，靠近路径规划图）
    save_path_3d = os.path.join(save_dir, f"{scene_name}_3d_view.png")
    _overlay_legend_on_image(temp_path_3d, legend_path, save_path_3d, 
                            position='top-right', margin=(150,200))  # 不再添加标题，已在Mayavi中生成
    
    # 裁剪白色边框
    _crop_white_borders(save_path_3d, save_path_3d)
    
    # 删除临时文件
    os.remove(temp_path_3d)
    os.remove(legend_path)
    
    print(f"✅ 3D 视图已保存: {save_path_3d}")
    
    mlab.close(fig_3d)

    # ==================== 2. 俯视图 ====================
    print(f"生成俯视图...")
    fig_top = mlab.figure('Top View', bgcolor=(1, 1, 1), size=(900, 900))
    
    # 绘制地形
    surf_top = _plot_terrain(H, map_size_x, map_size_y, 
                            colormap='summer', opacity=1.0)
    surf_top.module_manager.scalar_lut_manager.data_range = [50, terrain_max]
    surf_top.actor.actor.force_opaque = True
    
    # 添加 colorbar（右侧）
    colorbar_top = mlab.colorbar(surf_top, title='', orientation='vertical', nb_labels=None)
    # 设置 colorbar 位置（向左移动，靠近路径规划图）
    colorbar_top.scalar_bar_representation.position = [0.8, 0.25]  # 向左移到 0.86
    colorbar_top.scalar_bar_representation.position2 = [0.04, 0.35]  # 缩小一半
    # 设置标签数量和格式
    colorbar_top.scalar_bar.number_of_labels = num_labels
    colorbar_top.scalar_bar.label_format = '%.0f'
    # 设置标签样式（减小字体，与坐标轴数字一致）
    label_props_top = colorbar_top.label_text_property
    label_props_top.color = (0, 0, 0)
    label_props_top.font_size = 10  # 减小字体
    label_props_top.bold = False
    label_props_top.italic = False
    label_props_top.font_family = 'arial'
    # 设置 colorbar 标题样式
    title_props_top = colorbar_top.title_text_property
    title_props_top.font_size = 10
    title_props_top.bold = False
    title_props_top.italic = False
    title_props_top.font_family = 'arial'
    
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
    
    # 绘制路径
    z_offset = 3 if raise_topview else 0
    for i, path in enumerate(paths_absolute):
        _plot_path_line_and_midpoints(path, color_line=colors[i % len(colors)], 
                                      show_control_points=show_control_points,
                                      z_offset=z_offset)
    
    # 绘制起点和终点
    mlab.points3d(start_point_abs[0], start_point_abs[1], z_circle + z_offset, 
                 color=(0, 0, 0), scale_factor=14, mode='cube')
    mlab.points3d(end_point_abs[0], end_point_abs[1], z_circle + z_offset, 
                 color=(0, 0, 0), scale_factor=14, mode='sphere')
    
    # 绘制坐标轴
    _draw_custom_axes_and_grid(map_size_x, map_size_y, 0, 
                               draw_x=True, draw_y=True, draw_z=False)
    
    # 设置视角（模拟 View along +Z axis + Toggle parallel projection）
    # 俯视图：elevation=0（从上往下看XY平面）
    mlab.view(azimuth=0, elevation=0, distance='auto')
    
    # Toggle parallel projection: 启用正交投影
    fig_top.scene.camera.parallel_projection = True
    
    # 调整正交投影的缩放比例（parallel_scale越大，内容越小）
    # 减小parallel_scale以确保y轴刻度不被截断
    fig_top.scene.camera.parallel_scale = 750  # 从850减小到750，保留坐标轴标签显示空间
    
    # 在Mayavi场景中添加标题（使用text方法精确定位，单独放大字体）
    mlab.text(0.5, 0.1, '(b) Top view of the path', width=0.4, color=(0, 0, 0))
    
    # 保存图片（包含标题）
    temp_path_top = os.path.join(save_dir, f"{scene_name}_top_view_temp.png")
    mlab.savefig(temp_path_top, size=(800, 500), magnification=dpi/100)  # 与3D视图一致
    
    # 创建 matplotlib 图例（单列竖向排列）
    legend_path_top = os.path.join(save_dir, 'legend_temp_top.png')
    _create_matplotlib_legend(path_labels, colors, legend_path_top, ncol=1)  # ncol=1 竖着放
    
    # 叠加图例到俯视图（与3D视图位置一致）
    save_path_top = os.path.join(save_dir, f"{scene_name}_top_view.png")
    _overlay_legend_on_image(temp_path_top, legend_path_top, save_path_top,
                            position='top-right', margin=(300, 200))  # 不再添加标题，已在Mayavi中生成
    
    # 裁剪白色边框
    _crop_white_borders(save_path_top, save_path_top)
    
    # 删除临时文件
    os.remove(temp_path_top)
    os.remove(legend_path_top)
    
    print(f"✅ 俯视图已保存: {save_path_top}")
    
    mlab.close(fig_top)

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
 