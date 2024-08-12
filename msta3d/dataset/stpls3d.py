import torch
import numpy as np
import os.path as osp

from .scannetv2 import ScanNetDataset

class STPLS3DDataset(ScanNetDataset):

    CLASSES = (
        "building",
        "low vegetation",
        "med. vegetation",
        "high vegetation",
        "vehicle",
        "truck",
        "aircraft",
        "militaryVehicle",
        "bike",
        "motorcycle",
        "light pole",
        "street sign",
        "clutter",
        "fence",
    )

    NYU_ID = [i for i in range(20)]

    def load(self, filename):
        if self.with_label:
            xyz, rgb, superpoint_h, superpoints_l, semantic_label, instance_label = torch.load(filename)
            instance_label[semantic_label <= 0] = -100
            return xyz, rgb, superpoint_h, superpoints_l, semantic_label, instance_label
        else:
            xyz, rgb, superpoint_h, superpoints_l = torch.load(filename)
            dummy_sem_label = np.zeros(xyz.shape[0], dtype=np.float32)
            dummy_inst_label = np.zeros(xyz.shape[0], dtype=np.float32)
            return (
                xyz,
                rgb,
                superpoint_h,
                superpoints_l,
                dummy_sem_label,
                dummy_inst_label
                )

    def __getitem__(self, index: int):

        filename = self.filenames[index]
        scan_id = osp.basename(filename).replace(self.suffix, "")
        data = self.load(filename)
        data = (self.transform_train(*data) if self.training else self.transform_test(*data))
        (
            xyz,
            xyz_middle,
            rgb,
            superpoint_h,
            superpoint_l,
            semantic_label,
            instance_label,
        ) = data

        coord = torch.from_numpy(xyz).long()
        coord_float = torch.from_numpy(xyz_middle).float()
        feat = torch.from_numpy(rgb).float()

        superpoint_h = torch.from_numpy(superpoint_h)
        superpoint_l = torch.from_numpy(superpoint_l)

        semantic_label = torch.from_numpy(semantic_label).long()
        semantic_label = torch.where(semantic_label < 1, -100, semantic_label - 1)

        instance_label = torch.from_numpy(instance_label).long()
        inst = self.get_instance3D(coord_float, instance_label, semantic_label, superpoint_h)

        return scan_id, coord, coord_float, feat, superpoint_h, superpoint_l, inst