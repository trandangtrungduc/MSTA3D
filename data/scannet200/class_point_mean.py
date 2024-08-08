import glob
import torch
import numpy as np
from tqdm import tqdm

"""
Semantic_label = -100, 0, 1,...199
"""

file_path = sorted(glob.glob("train/*.pth") + glob.glob("val/*.pth"))
num_points = np.zeros(198)
num_insts = np.zeros(198)

progress_bar = tqdm(total=len(file_path))
for file in file_path:
    _, _, _, _, semantic_label, instance_label = torch.load(file)
    sem_unique = np.unique(semantic_label)
    for sem in sem_unique:
        if sem == -100 or sem == 0 or sem == 1:
            continue
        else:
            sem_id = sem - 2
            idx = np.argwhere(sem == semantic_label)
            num_point = len(idx)
            inst_unique = np.unique(instance_label[idx])
            num_inst = len(inst_unique)

            num_points[sem_id] = num_points[sem_id] + num_point
            num_insts[sem_id] = num_insts[sem_id] + num_inst

    progress_bar.update()
progress_bar.close()

class_mean_points = num_points / num_insts
print(class_mean_points)