import os
import json
import glob
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

COLOR_DETECTRON2 = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
    ]).astype(np.float32).reshape(-1, 3) * 255

def get_coords_colors(scene, spp):
    coords, colors, sem_label = scene[0], scene[1], scene[4]
    spp_label = scene[spp].astype(int)

    label_rgb = np.zeros(colors.shape)
    spp_num = spp_label.max() + 1
    spp_pointnum = np.zeros(spp_num)

    for idx in range(spp_num):
        spp_pointnum[idx] = (spp_label == idx).sum()
    sort_spp_idx = np.argsort(spp_pointnum)[::-1]

    for sort_id in range(spp_num):
        label_rgb[spp_label == sort_spp_idx[sort_id]] = COLOR_DETECTRON2[sort_id % len(COLOR_DETECTRON2)]
    colors = label_rgb

    sem_valid = (sem_label != -100)
    coords = coords[sem_valid]
    colors = colors[sem_valid]
    coords = coords[:, :3]
    colors = colors / 255
    return coords, colors

if __name__ == '__main__':
    spp_path = "data/scannetv2/"
    subset = "val"
    spp = 2 # 2: high-scale  3: low-scale
    scene_paths = glob.glob(spp_path + f"{subset}/*.pth")
    with open(f"tools/view_{subset}.json", "r") as file:
        views = json.load(file)
    file.close()

    for scene_path in scene_paths:
        room_name = scene_path[-29:-17]
        scene = torch.load(scene_path)
        coords, colors = get_coords_colors(scene, spp)

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(coords)
        pc.colors = o3d.utility.Vector3dVector(colors)

        R = pc.get_rotation_matrix_from_xyz(eval(views[room_name]))
        pc = pc.rotate(R, center=pc.get_center())

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pc)
        vis.get_render_option().point_size = 1.5
        vis.run()

        if not os.path.exists(f"output/{subset}/images/{room_name}"):
            os.mkdir(f"output/{subset}/images/{room_name}")
        vis.capture_screen_image(f"output/{subset}/images/{room_name}/{room_name}_{spp}.png")
        vis.destroy_window()


