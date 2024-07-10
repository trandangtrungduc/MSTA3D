import glob
import torch
import pickle
import shutil
import argparse
import numpy as np
import multiprocessing as mp

def get_args():
    parser = argparse.ArgumentParser('Create new dataset')
    parser.add_argument('--subset', type=str, default="train", help='train/val')
    parser.add_argument('--point_threshold', type=int, help='threshold to control number of points')
    args = parser.parse_args()
    return args

def create_new_data_set(fn):
    scene_name = fn[16:28] if args.subset == 'train' else fn[14:26]
    with open(fn, 'rb') as f:
        noisy_scan = pickle.load(f)

    # Calculate total number of noisy superpoint for each scan
    total = 0
    for _, point_id in noisy_scan.items():
        total += len(point_id[1])
    # print("Total noisy points: ", total)

    # Only create new dataset for scan less than point_threshold noisy points
    src = f'{args.subset}/{scene_name}_inst_nostuff.pth'
    if total > args.point_threshold:
        dest = f'filtered_{args.subset}_{args.point_threshold}/{scene_name}_inst_nostuff.pth'
        shutil.copyfile(src, dest)
    else:
        data = torch.load(src)
        coords, colors, superpoint_h, superpoint_l, sem_labels, instance_labels = data[0], data[1], data[2], data[3], data[4], data[5]
        # Only filtering on high-scale superpoints
        max_spp = superpoint_h.max()
        for spp_idx, p_idx in noisy_scan.items():
            spp_mask = np.where(superpoint_h == spp_idx)[0]
            spp_mask = spp_mask[p_idx[0]]
            for p_i in p_idx[1]:
                max_spp += 1
                superpoint_h[spp_mask[p_i]] = max_spp
        torch.save((coords, colors, superpoint_h, superpoint_l, sem_labels, instance_labels), f'filtered_{args.subset}_{args.point_threshold}/' + scene_name + '_inst_nostuff.pth')
    f.close()

if __name__ ==  '__main__':
    args = get_args()
    noisy_path = sorted(glob.glob(f"{args.subset}_noisy_spp/*.pickle"))

    p = mp.Pool(processes=mp.cpu_count())
    p.map(create_new_data_set, noisy_path)
    p.close()
    p.join()