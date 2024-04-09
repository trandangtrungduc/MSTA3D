import glob
import math
import numpy as np
import os.path as osp
import pointgroup_ops
import scipy.interpolate as interpolate
import scipy.ndimage as ndimage
import torch
import torch_scatter
from torch.utils.data import Dataset
from typing import Dict, Sequence, Tuple, Union

from ..utils import Instances3D

class ScanNetDataset(Dataset):
    CLASSES = (
        "cabinet",
        "bed",
        "chair",
        "sofa",
        "table",
        "door",
        "window",
        "bookshelf",
        "picture",
        "counter",
        "desk",
        "curtain",
        "refrigerator",
        "shower curtain",
        "toilet",
        "sink",
        "bathtub",
        "otherfurniture",
    )
    NYU_ID = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)

    def __init__(
        self,
        data_root,
        prefix,
        suffix,
        training=True,
        with_label=True,
        mode=4,
        with_elastic=True,
        use_xyz=True,
        repeat=1,
        voxel_cfg=None,
        logger=None,
    ):
        self.data_root = data_root
        self.prefix = prefix
        self.suffix = suffix
        self.voxel_cfg = voxel_cfg
        self.training = training
        self.with_label = with_label
        self.mode = mode
        self.with_elastic = with_elastic
        self.use_xyz = use_xyz
        self.repeat = repeat
        self.logger = logger
        self.filenames = self.get_filenames()
        self.logger.info(f"Load {self.prefix} dataset: {len(self.filenames)} scans.")

    def get_filenames(self):
        if self.prefix == "train_val":
            filenames_train = glob.glob(osp.join(self.data_root, "train", "*" + self.suffix))
            filenames_val = glob.glob(osp.join(self.data_root, "val", "*" + self.suffix))
            filenames = filenames_train + filenames_val
        else:
            filenames = glob.glob(
                osp.join(self.data_root, self.prefix, "*" + self.suffix)
            )
        assert len(filenames) > 0, "Empty dataset."
        filenames = sorted(filenames * self.repeat)
        return filenames

    def load(self, filename):
        if self.with_label:
            return torch.load(filename)
        else:
            xyz, rgb, superpoint_h, superpoint_l = torch.load(filename)
            dummy_sem_label = np.zeros(xyz.shape[0], dtype=np.float32)
            dummy_inst_label = np.zeros(xyz.shape[0], dtype=np.float32)
            return (
                xyz,
                rgb,
                superpoint_h,
                superpoint_l,
                dummy_sem_label,
                dummy_inst_label,
            )

    def __len__(self):
        return len(self.filenames)

    def __str__(self):
        return print(f"Data directory: ./{self.data_root}")

    def __getitem__(self, index: int) -> Tuple:
        filename = self.filenames[index]
        scan_id = osp.basename(filename).replace(self.suffix, "")
        data = self.load(filename)
        data = (
            self.transform_train(*data) if self.training else self.transform_test(*data)
        )
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
        semantic_label = torch.where(semantic_label < 2, -100, semantic_label - 2)
        instance_label = torch.from_numpy(instance_label).long()
        inst = self.get_instance3D(
            coord_float, instance_label, semantic_label, superpoint_h
        )
        return scan_id, coord, coord_float, feat, superpoint_h, superpoint_l, inst

    def transform_train(
        self, xyz, rgb, superpoint_h, superpoint_l, semantic_label, instance_label
    ):
        rgb = self.rgb_aug(
            rgb,
            apply=[True, True, False, False],
            noise=0.1,
            brightness=0.3,
            contrast=0.3,
            hsv=[0.2, 0.2, 0.2],
        )
        rgb = self.rgb_normalization(rgb)
        xyz_middle = self.data_aug(xyz, True, True, True)
        xyz_middle = self.coord_normalization(xyz_middle)
        xyz = xyz_middle * self.voxel_cfg.scale
        if self.with_elastic:
            xyz = self.elastic(xyz, 6, 40.0)
            xyz = self.elastic(xyz, 20, 160.0)
        xyz = xyz - xyz.min(0)
        xyz, valid_idxs = self.crop(xyz)
        xyz_middle = xyz_middle[valid_idxs]
        xyz = xyz[valid_idxs]
        rgb = rgb[valid_idxs]
        semantic_label = semantic_label[valid_idxs]
        superpoint_h = np.unique(superpoint_h[valid_idxs], return_inverse=True)[1]
        superpoint_l = np.unique(superpoint_l[valid_idxs], return_inverse=True)[1]
        instance_label = self.get_cropped_inst_label(instance_label, valid_idxs)
        return (
            xyz,
            xyz_middle,
            rgb,
            superpoint_h,
            superpoint_l,
            semantic_label,
            instance_label,
        )

    def transform_test(
        self, xyz, rgb, superpoint_h, superpoint_l, semantic_label, instance_label
    ):
        rgb = self.rgb_normalization(rgb)
        xyz_middle = xyz
        xyz_middle = self.coord_normalization(xyz_middle)
        xyz = xyz_middle * self.voxel_cfg.scale
        xyz -= xyz.min(0)
        valid_idxs = np.ones(xyz.shape[0], dtype=bool)
        superpoint_h = np.unique(superpoint_h[valid_idxs], return_inverse=True)[1]
        superpoint_l = np.unique(superpoint_l[valid_idxs], return_inverse=True)[1]
        instance_label = self.get_cropped_inst_label(instance_label, valid_idxs)
        return (
            xyz,
            xyz_middle,
            rgb,
            superpoint_h,
            superpoint_l,
            semantic_label,
            instance_label,
        )

    def coord_normalization(self, xyz):
        xyz = xyz - xyz.mean(0)
        return xyz

    def rgb_normalization(self, rgb):
        rgb = rgb / 127.5 - 1
        return rgb

    def data_aug(self, xyz, jitter=False, flip=False, rot=False):
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if flip:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(
                m,
                [
                    [math.cos(theta), math.sin(theta), 0],
                    [-math.sin(theta), math.cos(theta), 0],
                    [0, 0, 1],
                ],
            )
        return np.matmul(xyz, m)

    def rgb_aug(
        self,
        rgb,
        apply=[False, False, False, False],
        noise=0.1,
        brightness=0.3,
        contrast=0.3,
        hsv=[0.2, 0.2, 0.2],
    ):
        if apply[0]:
            rgb += np.random.randn(3) * noise

        if apply[1]:
            alpha = 1.0 + 2 * brightness * torch.rand(1) - brightness
            rgb *= alpha.item()

        if apply[2]:
            coef = np.array([[0.299, 0.587, 0.114]])
            beta = 1.0 + 2 * contrast * np.random.rand(1) - contrast
            gray = (rgb * coef).sum(axis=1)
            gray = (3.0 * (1.0 - beta) / gray.shape[0]) * np.sum(gray)
            rgb *= beta
            rgb += gray

        if apply[3]:
            h_shift_rad = hsv[0] * (np.pi / 180)
            vsu = hsv[1] * hsv[2] * np.cos(h_shift_rad)
            vsw = hsv[1] * hsv[2] * np.sin(h_shift_rad)

            ret_r = (
                (0.299 * hsv[2] + 0.701 * vsu + 0.168 * vsw) * rgb[:, 0]
                + (0.587 * hsv[2] - 0.587 * vsu + 0.330 * vsw) * rgb[:, 1]
                + (0.114 * hsv[2] - 0.114 * vsu - 0.497 * vsw) * rgb[:, 2]
            )

            ret_g = (
                (0.299 * hsv[2] - 0.299 * vsu - 0.328 * vsw) * rgb[:, 0]
                + (0.587 * hsv[2] + 0.413 * vsu + 0.035 * vsw) * rgb[:, 1]
                + (0.114 * hsv[2] - 0.114 * vsu + 0.292 * vsw) * rgb[:, 2]
            )

            ret_b = (
                (0.299 * hsv[2] - 0.300 * vsu + 1.25 * vsw) * rgb[:, 0]
                + (0.587 * hsv[2] - 0.588 * vsu - 1.05 * vsw) * rgb[:, 1]
                + (0.114 * hsv[2] + 0.886 * vsu - 0.203 * vsw) * rgb[:, 2]
            )
            rgb = np.stack((ret_r, ret_g, ret_b), axis=1)
        return rgb

    def crop(self, xyz: np.ndarray) -> Union[np.ndarray, np.ndarray]:
        xyz_offset = xyz.copy()
        valid_idxs = xyz_offset.min(1) >= 0
        assert valid_idxs.sum() == xyz.shape[0]

        full_scale = np.array([self.voxel_cfg.spatial_shape[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while valid_idxs.sum() > self.voxel_cfg.max_npoint:
            offset = np.clip(full_scale - room_range + 0.001, None, 0) * np.random.rand(
                3
            )
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * (
                (xyz_offset < full_scale).sum(1) == 3
            )
            full_scale[:2] -= 32

        return xyz_offset, valid_idxs

    def elastic(self, xyz, gran, mag):
        blur0 = np.ones((3, 1, 1)).astype("float32") / 3
        blur1 = np.ones((1, 3, 1)).astype("float32") / 3
        blur2 = np.ones((1, 1, 3)).astype("float32") / 3

        bb = np.abs(xyz).max(0).astype(np.int32) // gran + 3
        noise = [
            np.random.randn(bb[0], bb[1], bb[2]).astype("float32") for _ in range(3)
        ]
        noise = [
            ndimage.filters.convolve(n, blur0, mode="constant", cval=0) for n in noise
        ]
        noise = [
            ndimage.filters.convolve(n, blur1, mode="constant", cval=0) for n in noise
        ]
        noise = [
            ndimage.filters.convolve(n, blur2, mode="constant", cval=0) for n in noise
        ]
        noise = [
            ndimage.filters.convolve(n, blur0, mode="constant", cval=0) for n in noise
        ]
        noise = [
            ndimage.filters.convolve(n, blur1, mode="constant", cval=0) for n in noise
        ]
        noise = [
            ndimage.filters.convolve(n, blur2, mode="constant", cval=0) for n in noise
        ]
        ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
        interp = [
            interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0)
            for n in noise
        ]

        def g(xyz_):
            return np.hstack([i(xyz_)[:, None] for i in interp])

        return xyz + g(xyz) * mag

    def get_cropped_inst_label(
        self, instance_label: np.ndarray, valid_idxs: np.ndarray
    ) -> np.ndarray:
        instance_label = instance_label[valid_idxs]
        j = 0
        while j < instance_label.max():
            if len(np.where(instance_label == j)[0]) == 0:
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label

    def get_instance3D(self, coord_float, instance_label, semantic_label, superpoint_h):
        num_insts = instance_label.max().item() + 1
        num_points = len(instance_label)
        gt_masks, gt_labels, gt_boxes = [], [], []
        gt_inst = torch.zeros(num_points, dtype=torch.int64)
        for i in range(num_insts):
            idx = torch.where(instance_label == i)
            assert len(torch.unique(semantic_label[idx])) == 1
            sem_id = semantic_label[idx][0]
            if semantic_label[idx][0] == -100:
                continue

            gt_mask = torch.zeros(num_points)
            gt_mask[idx] = 1
            gt_masks.append(gt_mask)
            gt_label = sem_id
            gt_labels.append(gt_label)
            gt_box = torch.cat(
                (coord_float[idx].min(0)[0], coord_float[idx].max(0)[0]), dim=0
            )
            gt_boxes.append(gt_box)
            gt_inst[idx] = (sem_id + 1) * 1000 + i + 1
        if gt_masks:
            gt_masks = torch.stack(gt_masks, dim=0)
            gt_spmasks_h = torch_scatter.scatter_mean(
                gt_masks.float(), superpoint_h, dim=-1
            )
            gt_spmasks_h = (gt_spmasks_h > 0.5).float()
        else:
            gt_spmasks_h = torch.tensor([])
        if gt_boxes:
            gt_boxes = torch.stack(gt_boxes, dim=0)
        else:
            gt_boxes = torch.tensor([])
        gt_labels = torch.tensor(gt_labels)
        inst = Instances3D(num_points, gt_instances=gt_inst.numpy())
        inst.gt_labels = gt_labels.long()
        inst.gt_boxes = gt_boxes
        inst.gt_spmasks_h = gt_spmasks_h
        return inst

    def collate_fn(self, batch: Sequence[Sequence]) -> Dict:
        scan_ids, coords, coords_float, feats, superpoints_h, superpoints_l, insts = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        batch_offsets_h, batch_offsets_l = [0], [0]
        superpoint_h_bias, superpoint_l_bias = 0, 0

        for i, data in enumerate(batch):
            scan_id, coord, coord_float, feat, superpoint_h, superpoint_l, inst = data
            superpoint_h += superpoint_h_bias
            superpoint_h_bias = superpoint_h.max().item() + 1
            batch_offsets_h.append(superpoint_h_bias)
            superpoint_l += superpoint_l_bias
            superpoint_l_bias = superpoint_l.max().item() + 1
            batch_offsets_l.append(superpoint_l_bias)

            scan_ids.append(scan_id)
            coords.append(
                torch.cat([torch.LongTensor(coord.shape[0], 1).fill_(i), coord], 1)
            )
            coords_float.append(coord_float)
            feats.append(feat)
            superpoints_h.append(superpoint_h)
            superpoints_l.append(superpoint_l)
            insts.append(inst)

        batch_offsets_h = torch.tensor(batch_offsets_h, dtype=torch.int)
        batch_offsets_l = torch.tensor(batch_offsets_l, dtype=torch.int)
        coords = torch.cat(coords, 0)
        coords_float = torch.cat(coords_float, 0)
        feats = torch.cat(feats, 0)
        superpoints_h = torch.cat(superpoints_h, 0).long()
        superpoints_l = torch.cat(superpoints_l, 0).long()

        spatial_shape = np.clip(
            (coords.max(0)[0][1:] + 1).numpy(), self.voxel_cfg.spatial_shape[0], None
        )
        voxel_coords, v2p_map, p2v_map = pointgroup_ops.voxelization_idx(
            coords, len(batch), self.mode
        )
        return {
            "scan_ids": scan_ids,
            "voxel_coords": voxel_coords,
            "p2v_map": p2v_map,
            "v2p_map": v2p_map,
            "spatial_shape": spatial_shape,
            "coords_float": coords_float,
            "feats": feats,
            "superpoints_h": superpoints_h,
            "superpoints_l": superpoints_l,
            "batch_offsets_h": batch_offsets_h,
            "batch_offsets_l": batch_offsets_l,
            "insts": insts,
        }