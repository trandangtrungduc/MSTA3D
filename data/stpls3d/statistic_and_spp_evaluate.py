import glob
import argparse

import torch
import numpy as np
from tqdm import tqdm
from collections import Counter


def get_args():
    parser = argparse.ArgumentParser('ScanNet Exploration')
    parser.add_argument('--data_statistic', action='store_true', help='explore the ScanNet statistic')
    parser.add_argument('--method', type=str, help='which method to evaluate')
    parser.add_argument('--subset', type=str, default="train", help='train/val')
    args = parser.parse_args()
    return args

def computing_data_statistic(path, args):

    """ Function computing the data statistic for ScanNet """

    total_scans = len(path)
    number_of_points = []
    number_of_high_superpoint = []
    number_of_low_superpoint = []
    number_of_inst = []
    number_of_sem = []

    f = open(args.subset + "_statistic.txt", "a")
    print(f"Computing the statistic for {args.subset} set")
    progress_bar = tqdm(total=total_scans)
    for file in path:
        data = torch.load(file)
        num_of_points = data[0].shape[0]
        num_of_high_spp = data[2].max()
        num_of_low_spp = data[3].max()
        num_of_inst = data[5].max()
        num_of_sem = len(np.unique(data[4]))

        scene = file[15:-17] if args.subset == 'train' else file[13:-17]
        f.write(f"Scene: {scene}\nNumber of points: {num_of_points}\nNumber of high spp: {num_of_high_spp}\nNumber of low spp: {num_of_low_spp}\nNumber of instances: {num_of_inst}\nNumber of semantic labels: {num_of_sem}\n==============================\n")

        number_of_points.append(num_of_points)
        number_of_high_superpoint.append(num_of_high_spp)
        number_of_low_superpoint.append(num_of_low_spp)
        number_of_inst.append(num_of_inst)
        number_of_sem.append(num_of_sem)
        progress_bar.update()
    progress_bar.close()

    f.write(f"Total scans: {total_scans}\nAverage point clouds: {round(sum(number_of_points) / total_scans, 4)}\nAverage high superpoints: {round(sum(number_of_high_superpoint) / total_scans, 4)}\nAverage low superpoints: {round(sum(number_of_low_superpoint) / total_scans, 4)}\nAverage instances: {round(sum(number_of_inst) / total_scans, 4)}\nAverage semantic labels: {round(sum(number_of_sem) / total_scans, 4)}")
    f.close()

def accuracy_of_spg_method(path, args):

    """ Function evaluate the accuracy of the Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs (CVPR 2018) from https://github.com/l
    oicland/superpoint_graph """

    mean_high_acc_dataset, mean_low_acc_dataset = [], []
    total_scans = len(path)
    f = open(args.subset + "_spp_accuracy.txt", "a")
    print(f"Evaluating superpoint accuracy for {args.subset} set")
    progress_bar = tqdm(total=total_scans)
    for file in path:
        scene = file[15:-17] if args.subset == 'train' else file[13:-17]
        _, _, high_spp, low_spp, sem_label, _ = torch.load(file)
        unique_high_spp = np.unique(high_spp)
        unique_low_spp = np.unique(low_spp)
        mean_acc_high, mean_acc_low = [], []
        for spp in unique_high_spp:
            spp_id = np.where(spp == high_spp)[0]
            sem_of_points = sem_label[spp_id]
            label_counts = Counter(sem_of_points.tolist())
            most_common_label, _ = label_counts.most_common(1)[0]
            corrected_point = np.sum(most_common_label == sem_of_points)
            mean_acc_high.append(corrected_point * 100 / len(spp_id))
        for spp in unique_low_spp:
            spp_id = np.where(spp == low_spp)[0]
            sem_of_points = sem_label[spp_id]
            label_counts = Counter(sem_of_points.tolist())
            most_common_label, _ = label_counts.most_common(1)[0]
            corrected_point = np.sum(most_common_label == sem_of_points)
            mean_acc_low.append(corrected_point * 100 / len(spp_id))

        acc_high_spp = sum(mean_acc_high) / len(unique_high_spp)
        acc_low_spp = sum(mean_acc_low) / len(unique_low_spp)
        f.write(f"Scene: {scene}\nAverage scores high spp per scan: {round(acc_high_spp, 4)}\nAverage scores low spp per scan: {round(acc_low_spp, 4)}\n==========================================\n")
        mean_high_acc_dataset.append(acc_high_spp)
        mean_low_acc_dataset.append(acc_low_spp)
        progress_bar.update()
    progress_bar.close()
    f.write(f"Total scans: {total_scans}\nAverage high superpoint accuracy: {round(sum(mean_high_acc_dataset) / total_scans, 4)}\nAverage low superpoint accuracy: {round(sum(mean_low_acc_dataset) / total_scans, 4)}")
    f.close()

if __name__ == "__main__":

    train_path = sorted(glob.glob("train_with_spp/*.pth"))
    val_path = sorted(glob.glob("val_with_spp/*.pth"))

    args = get_args()
    if args.data_statistic:
        if args.subset == "train":
            computing_data_statistic(train_path, args)
        else:
            computing_data_statistic(val_path, args)

    if args.method == "spg":
        if args.subset == "train":
            accuracy_of_spg_method(train_path, args)
        else:
            accuracy_of_spg_method(val_path, args)

