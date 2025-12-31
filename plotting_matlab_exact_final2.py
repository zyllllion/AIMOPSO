#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-quality 3D path visualization script for paper
- Generates 3D view and top view only
- Saves high-resolution vector graphics directly
- Automatically adds legend and colorbar
"""

import numpy as np
from mayavi import mlab
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image


def _draw_cylinder(xc, yc, z_base, radius, height, color=(1, 0, 0), opacity=0.30, resolution=40):
    """Draw threat zone cylinder"""
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
    """Draw path line and control points"""
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
    """Draw terrain"""
    X, Y = np.meshgrid(np.arange(map_size_x), np.arange(map_size_y), indexing='xy')
    surf = mlab.mesh(X, Y, H, colormap=colormap, representation='surface', opacity=opacity)
    return surf


def _create_matplotlib_legend(labels, colors, save_path='legend_temp.png', ncol=1):
    """
    Create professional legend using matplotlib
    Args:
        labels: Algorithm name list
        colors: Color list (RGB tuple, 0-1 range)
        save_path: Legend save path
        ncol: Number of legend columns
    """
    fig = plt.figure(figsize=(3, 4), dpi=150)
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
                      fontsize=12,
                      handlelength=2.0,
                      handleheight=1.2,
                      columnspacing=1.5)
    
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_linewidth(1.0)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=150, 
                transparent=True, pad_inches=0.1)
    plt.close(fig)
    
    return save_path


def _overlay_legend_on_image(base_image_path, legend_path, output_path, 
                            position='top-right', margin=(50, 50)):
    """
    Overlay legend on image
    Args:
        base_image_path: Base image path
        legend_path: Legend image path
        output_path: Output image path
        position: Legend position
        margin: (x_margin, y_margin) margins
    """
    
    img_base = Image.open(base_image_path).convert('RGBA')
    img_legend = Image.open(legend_path).convert('RGBA')
    
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
    
    img_base.paste(img_legend, (x, y), img_legend)
    
    img_result = img_base.convert('RGB')
    img_result.save(output_path, quality=95)
    
    return output_path


def _crop_white_borders(image_path, output_path=None, border_threshold=250):
    """
    Crop white borders around image
    Args:
        image_path: Input image path
        output_path: Output image path (overwrites if None)
        border_threshold: White threshold (0-255)
    """
    from PIL import ImageChops
    
    img = Image.open(image_path).convert('RGB')
    
    bg = Image.new('RGB', img.size, (255, 255, 255))
    
    diff = ImageChops.difference(img, bg)
    
    diff = diff.convert('L')
    bbox = diff.getbbox()
    
    if bbox:
        img_cropped = img.crop(bbox)
        
        if output_path is None:
            output_path = image_path
        img_cropped.save(output_path, quality=95)
        
        return output_path
    else:
        return image_path


def _combine_images_horizontal(left_image_path, right_image_path, output_path, gap=20):
    """
    Combine two images horizontally
    Args:
        left_image_path: Left image path
        right_image_path: Right image path
        output_path: Output image path
        gap: Gap between images (pixels)
    
    Supported output formats:
        - .png: Saved using PIL (high quality)
        - .eps: Saved using matplotlib (vector format)
    """
    img_left = Image.open(left_image_path).convert('RGB')
    img_right = Image.open(right_image_path).convert('RGB')
    
    left_w, left_h = img_left.size
    right_w, right_h = img_right.size
    
    total_width = left_w + gap + right_w
    total_height = max(left_h, right_h)
    
    combined_img = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    
    left_y = (total_height - left_h) // 2
    combined_img.paste(img_left, (0, left_y))
    
    right_y = (total_height - right_h) // 2
    combined_img.paste(img_right, (left_w + gap, right_y))
    
    output_ext = os.path.splitext(output_path)[1].lower()
    
    if output_ext == '.eps':
        fig = plt.figure(figsize=(total_width / 100, total_height / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.imshow(combined_img)
        plt.savefig(output_path, format='eps', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"  ✅ EPS combined image saved")
    else:
        combined_img.save(output_path, quality=95)
    
    return output_path


def _draw_custom_axes_and_grid(map_size_x, map_size_y, scene_z_max,
                               draw_x=True, draw_y=True, draw_z=True,
                               yz_grid_at_far_side=False):
    """Draw custom axes and grid"""
    grid_color = (0.7, 0.7, 0.7)
    axis_color = (0, 0, 0)
    label_color = (0, 0, 0)
    tick_length = 20
    tick_interval = 200

    if draw_x and draw_y:
        for x_line in np.arange(0, map_size_x + 1, tick_interval):
            mlab.plot3d([x_line, x_line], [0, map_size_y], [0, 0], 
                       color=grid_color, tube_radius=0.5)
        for y_line in np.arange(0, map_size_y + 1, tick_interval):
            mlab.plot3d([0, map_size_x], [y_line, y_line], [0, 0], 
                       color=grid_color, tube_radius=0.5)
    
    if draw_y and draw_z:
        grid_x_pos = map_size_x if yz_grid_at_far_side else 0
        for y_line in np.arange(0, map_size_y + 1, tick_interval):
            mlab.plot3d([grid_x_pos, grid_x_pos], [y_line, y_line], 
                       [0, scene_z_max], color=grid_color, tube_radius=0.5)
        for z_line in np.arange(0, scene_z_max + 1, 100):
            mlab.plot3d([grid_x_pos, grid_x_pos], [0, map_size_y], 
                       [z_line, z_line], color=grid_color, tube_radius=0.5)
    
    if draw_x and draw_z:
        for x_line in np.arange(0, map_size_x + 1, tick_interval):
            mlab.plot3d([x_line, x_line], [map_size_y, map_size_y], 
                       [0, scene_z_max], color=grid_color, tube_radius=0.5)
        for z_line in np.arange(0, scene_z_max + 1, 100):
            mlab.plot3d([0, map_size_x], [map_size_y, map_size_y], 
                       [z_line, z_line], color=grid_color, tube_radius=0.5)

    if draw_x:
        mlab.plot3d([0, map_size_x], [0, 0], [0, 0], color=axis_color, tube_radius=1.5)
        mlab.text3d(map_size_x / 2, -tick_length * 4, 0, 'x [m]', 
                   scale=15, color=label_color)
        for x in np.arange(0, map_size_x + 1, tick_interval):
            mlab.plot3d([x, x], [0, -tick_length], [0, 0], 
                       color=axis_color, tube_radius=1)
            mlab.text3d(x, -tick_length * 2.5, 0, str(x), 
                       scale=12, color=label_color)
    
    if draw_y:
        mlab.plot3d([0, 0], [0, map_size_y], [0, 0], color=axis_color, tube_radius=1.5)
        mlab.text3d(-tick_length * 7, map_size_y / 2, 0, 'y [m]', 
                   scale=15, color=label_color)
        for y in np.arange(0, map_size_y + 1, tick_interval):
            mlab.plot3d([-tick_length, 0], [y, y], [0, 0], 
                       color=axis_color, tube_radius=1)
            mlab.text3d(-tick_length * 3.5, y, 0, str(y), 
                       scale=12, color=label_color)
    
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
    Generate and save high-quality 3D visualization for paper
    
    Args:
        paths_absolute: Path list
        path_labels: Algorithm name list
        experiment_group: Experiment group number (optional)
        model: Environment model dict
        save_dir: Save directory
        scene_name: Scene name
        show_control_points: Whether to show control points
        raise_topview: Whether to raise paths in top view
        dpi: Image resolution
    """
    H = model['H']
    threats = model.get('threats', np.zeros((0, 4)))
    map_size_x, map_size_y = model['map_range']

    scene_z_max = 400
    cyl_height = 400

    colors = [
        (1, 0, 1),
        (0, 0, 1),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 1),
        (0.5, 0, 0.5),
        (1, 0.5, 0),
        (0, 0, 0)
    ]

    if not paths_absolute:
        print("Warning: No paths to plot.")
        return
    if experiment_group is not None:
        if experiment_group == 1:
            save_dir = os.path.join(save_dir, "pso_variants")
        elif experiment_group == 2:
            save_dir = os.path.join(save_dir, "classic_algorithms")
        else:
            save_dir = os.path.join(save_dir, str(experiment_group))
    os.makedirs(save_dir, exist_ok=True)

    terrain_max = np.ceil(H.max() / 50) * 50
    
    aimopso_idx = None
    for i, label in enumerate(path_labels):
        if 'A*IMOPSO' in label or 'A*MOPSO' in label or 'AIMOPSO' in label:
            aimopso_idx = i
            break
    if aimopso_idx is not None and aimopso_idx != 0:
        path_labels_reordered = [path_labels[aimopso_idx]] + path_labels[:aimopso_idx] + path_labels[aimopso_idx+1:]
        colors_reordered = [colors[aimopso_idx]] + colors[:aimopso_idx] + colors[aimopso_idx+1:]
        paths_absolute_reordered = [paths_absolute[aimopso_idx]] + paths_absolute[:aimopso_idx] + paths_absolute[aimopso_idx+1:]
        
        path_labels = path_labels_reordered
        colors = colors_reordered
        paths_absolute = paths_absolute_reordered

    print(f"Generating 3D view...")
    fig_3d = mlab.figure('3D View', bgcolor=(1, 1, 1), size=(1200, 900))
    
    terrain_surface = _plot_terrain(H, map_size_x, map_size_y, 
                                    colormap='summer', opacity=1.0)
    terrain_surface.module_manager.scalar_lut_manager.data_range = [50, terrain_max]
    
    colorbar = mlab.colorbar(terrain_surface, title='', orientation='vertical', nb_labels=None)
    colorbar.scalar_bar_representation.position = [0.86, 0.25]
    colorbar.scalar_bar_representation.position2 = [0.04, 0.35]
    num_labels = int((terrain_max - 50) / 50) + 1
    colorbar.scalar_bar.number_of_labels = num_labels
    colorbar.scalar_bar.label_format = '%.0f'
    label_props = colorbar.label_text_property
    label_props.color = (0, 0, 0)
    label_props.font_size = 10
    label_props.bold = False
    label_props.italic = False
    label_props.font_family = 'arial'
    title_props = colorbar.title_text_property
    title_props.font_size = 10
    title_props.bold = False
    title_props.italic = False
    title_props.font_family = 'arial'
    
    _draw_custom_axes_and_grid(map_size_x, map_size_y, scene_z_max,
                               draw_x=True, draw_y=True, draw_z=True,
                               yz_grid_at_far_side=True)
    
    for (x0, y0, z0, R) in threats:
        _draw_cylinder(x0, y0, z0, R, cyl_height, opacity=0.30)
    for i, path in enumerate(paths_absolute):
        _plot_path_line_and_midpoints(path, color_line=colors[i % len(colors)], 
                                      show_control_points=show_control_points)
    
    start_point_abs = paths_absolute[0][0, :]
    end_point_abs = paths_absolute[0][-1, :]
    mlab.points3d(start_point_abs[0], start_point_abs[1], start_point_abs[2], 
                 color=(0, 0, 0), scale_factor=14, mode='cube')
    mlab.points3d(end_point_abs[0], end_point_abs[1], end_point_abs[2], 
                 color=(0, 0, 0), scale_factor=14, mode='sphere')
    
    mlab.view(azimuth=-135, elevation=65, distance='auto')
    
    mlab.text(0.5, 0.1, '(a) 3D path view', width=0.3, color=(0, 0, 0))
    
    temp_path_3d = os.path.join(save_dir, f"{scene_name}_3d_view_temp.png")
    mlab.savefig(temp_path_3d, size=(800, 500), magnification=dpi/100)
    

    legend_path = os.path.join(save_dir, 'legend_temp.png')
    _create_matplotlib_legend(path_labels, colors, legend_path, ncol=1)
    
    save_path_3d = os.path.join(save_dir, f"{scene_name}_3d_view.png")
    _overlay_legend_on_image(temp_path_3d, legend_path, save_path_3d, 
                            position='top-right', margin=(150,200))
    
    _crop_white_borders(save_path_3d, save_path_3d)
    
    os.remove(temp_path_3d)
    os.remove(legend_path)
    
    print(f"✅ 3D view saved: {save_path_3d}")
    
    mlab.close(fig_3d)
    
    fig_top = mlab.figure('Top View', bgcolor=(1, 1, 1), size=(900, 900))
    
    surf_top = _plot_terrain(H, map_size_x, map_size_y, 
                            colormap='summer', opacity=1.0)
    surf_top.module_manager.scalar_lut_manager.data_range = [50, terrain_max]
    surf_top.actor.actor.force_opaque = True
    
    colorbar_top = mlab.colorbar(surf_top, title='', orientation='vertical', nb_labels=None)
    colorbar_top.scalar_bar_representation.position = [0.8, 0.25]
    colorbar_top.scalar_bar_representation.position2 = [0.04, 0.35]
    colorbar_top.scalar_bar.number_of_labels = num_labels
    colorbar_top.scalar_bar.label_format = '%.0f'
    label_props_top = colorbar_top.label_text_property
    label_props_top.color = (0, 0, 0)
    label_props_top.font_size = 10
    label_props_top.bold = False
    label_props_top.italic = False
    label_props_top.font_family = 'arial'
    title_props_top = colorbar_top.title_text_property
    title_props_top.font_size = 10
    title_props_top.bold = False
    title_props_top.italic = False
    title_props_top.font_family = 'arial'
    
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
    
    z_offset = 3 if raise_topview else 0
    for i, path in enumerate(paths_absolute):
        _plot_path_line_and_midpoints(path, color_line=colors[i % len(colors)], 
                                      show_control_points=show_control_points,
                                      z_offset=z_offset)
    
    mlab.points3d(start_point_abs[0], start_point_abs[1], z_circle + z_offset, 
                 color=(0, 0, 0), scale_factor=14, mode='cube')
    mlab.points3d(end_point_abs[0], end_point_abs[1], z_circle + z_offset, 
                 color=(0, 0, 0), scale_factor=14, mode='sphere')
    
    _draw_custom_axes_and_grid(map_size_x, map_size_y, 0, 
                               draw_x=True, draw_y=True, draw_z=False)
    
    mlab.view(azimuth=0, elevation=0, distance='auto')
    
    fig_top.scene.camera.parallel_projection = True
    
    fig_top.scene.camera.parallel_scale = 750
    
    mlab.text(0.5, 0.1, '(b) Top view of the path', width=0.4, color=(0, 0, 0))
    
    temp_path_top = os.path.join(save_dir, f"{scene_name}_top_view_temp.png")
    mlab.savefig(temp_path_top, size=(800, 500), magnification=dpi/100)
    
    legend_path_top = os.path.join(save_dir, 'legend_temp_top.png')
    _create_matplotlib_legend(path_labels, colors, legend_path_top, ncol=1)
    
    save_path_top = os.path.join(save_dir, f"{scene_name}_top_view.png")
    _overlay_legend_on_image(temp_path_top, legend_path_top, save_path_top,
                            position='top-right', margin=(300, 200))
    
    _crop_white_borders(save_path_top, save_path_top)
    
    os.remove(temp_path_top)
    os.remove(legend_path_top)
    
    print(f"✅ Top view saved: {save_path_top}")
    
    mlab.close(fig_top)
    
    combined_path_png = os.path.join(save_dir, f"{scene_name}_combined.png")
    _combine_images_horizontal(save_path_3d, save_path_top, combined_path_png, gap=3)
    print(f"✅ PNG combined image saved: {combined_path_png}")
    
    combined_path_eps = os.path.join(save_dir, f"{scene_name}_combined.eps")
    _combine_images_horizontal(save_path_3d, save_path_top, combined_path_eps, gap=3)
    print(f"✅ EPS combined image saved: {combined_path_eps}")

    print(f"\n✅ All images saved to: {save_dir}")
    print(f"   - {scene_name}_3d_view.png (3D view)")
    print(f"   - {scene_name}_top_view.png (Top view)")
    print(f"   - {scene_name}_combined.png (PNG combined)")
    print(f"   - {scene_name}_combined.eps (EPS combined) ⭐")


if __name__ == "__main__":
    print("This module generates high-quality 3D visualizations for papers.")
    print("Please call from compare_algorithms_visual_cached.py.")
 
