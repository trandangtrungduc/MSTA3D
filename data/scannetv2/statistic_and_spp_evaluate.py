import glob
import argparse

import torch
import numpy as np
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
    number_of_superpoints = []
    number_of_inst = []
    number_of_sem = []

    f = open(args.subset + "_statistic.txt", "a")
    for file in path:
        data = torch.load(file)
        num_of_points = data[0].shape[0]
        num_of_superpoints = data[2].max() # 2 for high-scale, 3 for low-scale
        num_of_inst = data[5].max()
        num_of_sem = len(np.unique(data[4]))

        scene = file[6:18] if args.subset == 'train' else file[4:16]
        f.write(f"Scene: {scene}\nNumber of points: {num_of_points}\nNumber of superpoints: {num_of_superpoints}\nNumber of instances: {num_of_inst}\nNumber of semantic labels: {num_of_sem}\n==============================\n")

        number_of_points.append(num_of_points)
        number_of_superpoints.append(num_of_superpoints)
        number_of_inst.append(num_of_inst)
        number_of_sem.append(num_of_sem)

    f.write(f"Total scans: {total_scans}\nAverage point clouds: {round(sum(number_of_points) / total_scans, 2)}\nAverage superpoints: {round(sum(number_of_superpoints) / total_scans, 2)}\nAverage instances: {round(sum(number_of_inst) / total_scans, 2)}\nAverage semantic labels: {round(sum(number_of_sem) / total_scans, 2)}")
    f.close()

def accuracy_of_method_1(path, args):

    """ Function evaluate the accuracy of the Graph Based Image Segmentation (IJCV 2004) from https://github.com/ScanNet/ScanNet/tree/master/Segmentator """

    mean_accuracy_of_dataset = []
    total_scans = len(path)
    f = open(args.subset + "_superpoint_accuracy.txt", "a")
    for file in path:
        scene = file[6:18] if args.subset == 'train' else file[4:16]
        file = torch.load(file)
        superpoints, sem_label = file[2], file[4]
        unique_superpoints = np.unique(superpoints)
        mean_accuracy_of_scene = []
        for superpoint in unique_superpoints:
            spp_id = np.where(superpoint == superpoints)[0]
            sem_of_points = sem_label[spp_id]
            label_counts = Counter(sem_of_points.tolist())
            most_common_label, _ = label_counts.most_common(1)[0]
            corrected_point = np.sum(most_common_label ==  sem_of_points)
            mean_accuracy_of_scene.append(corrected_point * 100 / len(spp_id))
        acc_spp = sum(mean_accuracy_of_scene) / len(unique_superpoints)
        f.write(f"Scene: {scene}\nAverage scores per scan: {round(acc_spp,2)}\n==============================\n")
        mean_accuracy_of_dataset.append(acc_spp)
    f.write(f"Total scans: {total_scans}\nAverage superpoint accuracy: {round(sum(mean_accuracy_of_dataset) / total_scans, 2)}")
    f.close()

def accuracy_of_method_2(path, scannet_path, args):

    """ Function evaluate the accuracy of the Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs (CVPR 2018) from https://github.com/l
    oicland/superpoint_graph """

    path = sorted(glob.glob(path + "low_scale/*.pth")) # low-scale
    mean_accuracy_of_dataset = []
    total_scans = len(path)
    f = open(args.subset + "_low_spp_acc_method_2.txt", "a")
    for file, sem_path in zip(path, scannet_path):
        scene = sem_path[6:18] # file[4:16] for low-scale
        superpoints = torch.load(file)
        sem_label = torch.load(sem_path)
        sem_label = sem_label[4]
        unique_superpoints = np.unique(superpoints)
        mean_accuracy_of_scene = []
        for superpoint in unique_superpoints:
            spp_id = np.where(superpoint == superpoints)[0]
            sem_of_points = sem_label[spp_id]
            label_counts = Counter(sem_of_points.tolist())
            most_common_label, _ = label_counts.most_common(1)[0]
            corrected_point = np.sum(most_common_label == sem_of_points)
            mean_accuracy_of_scene.append(corrected_point * 100 / len(spp_id))
        acc_spp = sum(mean_accuracy_of_scene) / len(unique_superpoints)
        f.write(f"Scene: {scene}\nAverage scores per scan: {round(acc_spp, 2)}\n==============================\n")
        mean_accuracy_of_dataset.append(acc_spp)
    f.write(f"Total scans: {total_scans}\nAverage superpoint accuracy: {round(sum(mean_accuracy_of_dataset) / total_scans, 2)}")
    f.close()


def accuracy_of_method_3(path):

    """ Function evaluate the accuracy of the Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs (ICCV 2023) from https://github.com/drprojects/superpoint_transformer """

    return None

if __name__ == "__main__":

    class_numpoint_mean = [-100., -100., 3917., 12056., 2303.,
                          8331., 3948., 3166., 5629., 11719.,
                          1003., 3317., 4912., 10221., 3889.,
                          4136., 2120., 945., 3967., 2589.]
    semantic_label_names = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookspg_train_pathshelf',
    'picture', 'counter', 'desk', 'curtain','refrigerator',
    'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
    class_sem = dict(map(lambda i,j : (i,j) , semantic_label_names,class_numpoint_mean))

    print("======== Average number of points each class ========")
    for key, value in class_sem.items():
        print(f"{key}: {value} points \n")
    print("=" * 50)

    train_path = sorted(glob.glob("train/*.pth"))
    val_path = sorted(glob.glob("val/*.pth"))

    args = get_args()
    if args.data_statistic:
        if args.subset == "train":
            computing_data_statistic(train_path, args)
        else:
            computing_data_statistic(val_path, args)

    if args.method == "1":
        if args.subset == "train":
            accuracy_of_method_1(train_path, args)
        else:
            accuracy_of_method_1(val_path, args)

    if args.method == "2":
        spg_train_path = "spp_from_spp_graph/train/"
        spg_val_path = "spp_from_spp_graph/val/"
        if args.subset == "train":
            accuracy_of_method_2(spg_train_path, train_path, args)
        else:
            accuracy_of_method_2(spg_val_path, val_path, args)

    # elif args.method == "3":
    #     if args.subset == "train":
    #         accuracy_of_method_3(train_path, args)
    #     else:
    #         accuracy_of_method_3(val_path, args)
    # else:
        # raise NotImplementedError

