import os
import glob
import torch
import argparse
import segmentator
import numpy as np
import open3d as o3d
import multiprocessing as mp

def spp_process(file):
    """ This implementation for different scale superpoints.
        S0: kThresh=0.01, segMinVerts=20
        S1: kThresh=0.1, segMinVerts=40
    """
    new_data = {}
    data = torch.load(file)
    scene_id = data["scene_id"]
    new_data["scene_id"] = scene_id

    mesh = o3d.io.read_triangle_mesh(f"{config.data_split}_ply/" + scene_id + "_mesh_aligned_0.05.ply")
    vertices = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))
    faces = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))
    superpoint_h = segmentator.segment_mesh(vertices, faces, kThresh=0.01, segMinVerts=20).numpy()
    superpoint_l = segmentator.segment_mesh(vertices, faces, kThresh=0.1, segMinVerts=40).numpy()

    new_data["sampled_coords"] = data["sampled_coords"]
    new_data["sampled_colors"] = data["sampled_colors"]
    new_data["sampled_num_labels"] = data["sampled_num_labels"]
    new_data["sampled_labels"] = data["sampled_labels"]
    new_data["sampled_instance_labels"] = data["sampled_instance_labels"]
    new_data["sampled_instance_anno_id"] = data["sampled_instance_anno_id"]
    new_data["superpoint_h"] = superpoint_h
    new_data["superpoint_l"] = superpoint_l

    torch.save(new_data, f"{config.data_split}_with_spp/"  + scene_id + ".pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_split', help='data split (train / val)', default='train')
    config = parser.parse_args()

    files = sorted(glob.glob(config.data_split + '/*.pth'))
    p = mp.Pool(processes=mp.cpu_count())
    p.map(spp_process, files)
    p.close()
    p.join()