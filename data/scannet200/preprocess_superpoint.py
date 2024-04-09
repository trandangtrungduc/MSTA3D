import os
import glob
import torch
import argparse
import segmentator
import numpy as np
import open3d as o3d
import multiprocessing as mp

def spp_process(fn):
    """ This implementation for different scale superpoints.
        S0: kThresh=0.01, segMinVerts=20
        S1: kThresh=0.1, segMinVerts=40
    """
    mesh = o3d.io.read_triangle_mesh(fn)
    vertices = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))
    faces = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))
    superpoint_h = segmentator.segment_mesh(vertices, faces, kThresh=0.01, segMinVerts=20).numpy()
    superpoint_l = segmentator.segment_mesh(vertices, faces, kThresh=0.1, segMinVerts=40).numpy()

    coord, color, sem_labels, inst_label = torch.load(config.data_split + "/" + fn[-27:-15] + "_inst_nostuff.pth")
    torch.save(
        (coord, color, superpoint_h, superpoint_l, sem_labels, inst_label),
        os.path.join(config.data_split, f"{fn[-27:-15]}_inst_nostuff.pth"),
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True, help="Path to the ScanNet dataset containing scene folders")
    parser.add_argument("--output_root", required=True, help="Output path where train/val/test folders will be located")
    parser.add_argument('--data_split', help='data split (train / val)', default='train')
    config = parser.parse_args()

    files = sorted(glob.glob(config.dataset_root + "/" + config.data_split + '/*_vh_clean_2.ply'))

    p = mp.Pool(processes=mp.cpu_count())
    p.map(spp_process, files)
    p.close()
    p.join()
