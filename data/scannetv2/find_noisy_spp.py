import glob
import pickle
import argparse
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

import torch
import torch.nn.functional as F

def get_args():
    parser = argparse.ArgumentParser('Find noisy superpoint')
    parser.add_argument('--subset', type=str, default="train", help='train/val')
    parser.add_argument('--sim_threshold', type=float, help='threshold for cosine similarity')
    args = parser.parse_args()
    return args

def two_non_empty_lists(lst1, lst2):
    non_empty_count = sum([bool(lst) for lst in [lst1, lst2]])
    return non_empty_count == 2

def find_outliers(raw_feats):
    outliers = []
    for col in range(raw_feats.shape[1]):
        col_data = raw_feats[:, col]
        q1 = np.quantile(col_data, 0.25)
        q3 = np.quantile(col_data, 0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_indices = np.where((col_data < lower_bound) | (col_data > upper_bound))[0]
        outliers.append(outlier_indices.tolist())
    outliers = [item for sublist in outliers for item in sublist]
    return list(set(outliers))

def find_similarity(feats, threshold):
    feats = torch.from_numpy(feats)
    normalized_tensor = F.normalize(feats, p=2, dim=1)
    cosine_similarity_matrix = torch.matmul(normalized_tensor, normalized_tensor.T)
    indices = torch.where((cosine_similarity_matrix < threshold) & (cosine_similarity_matrix != 1))
    filtered_indices = [(i.item(), j.item()) for i, j in zip(indices[0], indices[1]) if i < j]
    filtered_indices = list(set(item for sublist in filtered_indices for item in sublist))
    if len(filtered_indices) != 0:
        return filtered_indices
    return None

def find_noisy_spp(fn):
    scene_name = fn[6:18] if args.subset == 'train' else fn[4:16]
    coords, colors, superpoint_h, _, _, _ = torch.load(fn)
    superpoint_features = torch.load(f'point_{args.subset}_feats/{scene_name}_inst_nostuff.pth')

    noisy_sppoint = {}
    superpoint_labels = np.unique(superpoint_h)
    for spp_id in superpoint_labels:
        spp_mask = np.where(superpoint_h == spp_id)[0]
        spp_coords = coords[spp_mask]
        spp_colors = colors[spp_mask]

        # Filtering using quartile
        filered_coords_idx = find_outliers(spp_coords)
        filered_colors_idx = find_outliers(spp_colors)

        # Further filtering with point feature similarity
        if two_non_empty_lists(filered_coords_idx, filered_colors_idx):
            spp_features = superpoint_features[spp_mask]
            unique_point_idx = [item for item in filered_coords_idx if item in filered_colors_idx]
            filtered_feat_idx = find_similarity(spp_features[unique_point_idx], args.sim_threshold)
            if filtered_feat_idx is not None:
                noisy_sppoint[spp_id] = (unique_point_idx, filtered_feat_idx)
        elif filered_coords_idx:
            spp_features = superpoint_features[spp_mask]
            filtered_feat_idx = find_similarity(spp_features[filered_coords_idx], args.sim_threshold)
            if filtered_feat_idx is not None:
                noisy_sppoint[spp_id] = (filered_coords_idx, filtered_feat_idx)
        elif filered_colors_idx:
            spp_features = superpoint_features[spp_mask]
            filtered_feat_idx = find_similarity(spp_features[filered_colors_idx], args.sim_threshold)
            if filtered_feat_idx is not None:
                noisy_sppoint[spp_id] = (filered_colors_idx, filtered_feat_idx)
        else:
            continue
    if len(noisy_sppoint) != 0:
        with open(f'{args.subset}_noisy_spp/{scene_name}_inst_nostuff.pickle', 'wb') as f:
            pickle.dump(noisy_sppoint, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

if __name__ ==  '__main__':
    args = get_args()
    data_path = sorted(glob.glob(f"{args.subset}/*.pth"))

    p = mp.Pool(processes=mp.cpu_count())
    p.map(find_noisy_spp, data_path)
    p.close()
    p.join()