import gorilla
import functools
import pointgroup_ops
import numpy as np
import spconv.pytorch as spconv
from sklearn.cluster import DBSCAN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from .loss import Criterion
from .query_decoder import QueryDecoder
from .backbone import ResidualBlock, UBlock
from msta3d.utils import cuda_cast, rle_encode

@gorilla.MODELS.register_module()
class MSTA3D(nn.Module):
    def __init__(self,
                 input_channel: int = 6,
                 blocks: int = 5,
                 block_reps: int = 2,
                 media: int = 32,
                 normalize_before=True,
                 return_blocks=True,
                 num_class=18,
                 decoder=None,
                 criterion=None,
                 test_cfg=None,
                 norm_eval=False,
                 iou_threshold=0.9,
                 dbscan_eps=1.0,
                 fix_module=[]):

        super().__init__()
        self.input_conv = spconv.SparseSequential(spconv.SubMConv3d(input_channel,  media, kernel_size=3, padding=1, bias=False, indice_key="subm1"))
        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        block_list = [media * (i + 1) for i in range(blocks)]
        self.unet = UBlock( block_list, norm_fn, block_reps, block, indice_key_id=1, normalize_before=normalize_before, return_blocks=return_blocks,
        )

        self.output_layer = spconv.SparseSequential(norm_fn(media), nn.ReLU(inplace=True))
        self.num_class = num_class
        self.decoder = QueryDecoder(**decoder, in_channel=media, num_class=num_class)
        self.criterion = Criterion(**criterion, num_class=num_class)
        self.test_cfg = test_cfg
        self.norm_eval = norm_eval
        self.iou_threshold = iou_threshold
        self.dbscan_eps = dbscan_eps

        for module in fix_module:
            module = getattr(self, module)
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(MSTA3D, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    def forward(self, batch, mode="loss"):
        if mode == "loss":
            return self.loss(**batch)
        elif mode == "predict":
            return self.predict(**batch)

    @cuda_cast
    def loss(self, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape, coords_float, feats, superpoints_h, superpoints_l, batch_offsets_h, batch_offsets_l, insts):

        batch_size = len(batch_offsets_h) - 1
        feats = torch.cat((feats, coords_float), dim=1)
        voxel_feats = pointgroup_ops.voxelization(feats, p2v_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

        sp_feats_h, sp_feats_l = self.extract_feat(input, superpoints_h, superpoints_l, v2p_map)

        out = self.decoder(sp_feats_h, sp_feats_l, batch_offsets_h, batch_offsets_l)

        loss, loss_dict = self.criterion(out, insts)
        return loss, loss_dict

    @cuda_cast
    def predict(self, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape, coords_float, feats, superpoints_h, superpoints_l, batch_offsets_h, batch_offsets_l, insts):

        batch_size = len(batch_offsets_h) - 1
        feats = torch.cat((feats, coords_float), dim=1)
        voxel_feats = pointgroup_ops.voxelization(feats, p2v_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

        sp_feats_h, sp_feats_l = self.extract_feat(input, superpoints_h, superpoints_l, v2p_map)

        out = self.decoder(sp_feats_h, sp_feats_l, batch_offsets_h, batch_offsets_l)

        ret = self.predict_by_feat(scan_ids, out, superpoints_h, coords_float, insts)
        return ret

    def get_refined_mask(self, mask_pred, boxes_pred, coords_float, pred_box_scores):
        for i in range(mask_pred.shape[0]):
            if pred_box_scores[i] >= self.iou_threshold:
                filtered_mask = torch.zeros_like(mask_pred[i])
                mask_idx = torch.where(mask_pred[i] == 1)[0]
                mask_coords = coords_float[mask_idx]
                in_box_mask = torch.all(torch.logical_and(mask_coords >= boxes_pred[i][:3], mask_coords <= boxes_pred[i][3:6]), dim=-1)
                filtered_idx = torch.where(in_box_mask == True)[0]
                filtered_mask[mask_idx[filtered_idx]] = 1
                mask_pred[i] = filtered_mask
        return mask_pred

    def predict_by_feat(self, scan_ids, decoder_output, superpoints_h, coords_float, insts):

        pred_labels = decoder_output["labels"]
        pred_masks = decoder_output["masks"][0]
        pred_scores = decoder_output["mask_scores"][0]
        pred_box_scores = decoder_output["box_scores"][0]

        scores = F.softmax(pred_labels[0], dim=-1)[:, :-1]
        scores *= pred_scores
        scores *= pred_box_scores

        labels = (torch.arange(self.num_class, device=scores.device).unsqueeze(0).repeat(self.decoder.num_query, 1).flatten(0, 1))
        scores, topk_idx = scores.flatten(0, 1).topk(self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]
        labels += 1
        topk_idx = torch.div(topk_idx, self.num_class, rounding_mode="floor")
        mask_pred = pred_masks[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        mask_pred = (mask_pred > 0).float()
        mask_scores = (mask_pred_sigmoid * mask_pred).sum(1) / (mask_pred.sum(1) + 1e-6)
        scores = scores * mask_scores

        mask_pred = mask_pred[:, superpoints_h].int()

        if self.iou_threshold != 0:
            pred_boxes = decoder_output["boxes"][0]
            boxes_pred = pred_boxes[topk_idx]
            mask_pred = self.get_refined_mask(mask_pred, boxes_pred, coords_float, pred_box_scores)

        if self.dbscan_eps != 0:
            mask_pred = self.dbscan_post_processing(coords_float, mask_pred)

        score_mask = scores > self.test_cfg.score_thr
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        cls_pred = labels.cpu().numpy()
        score_pred = scores.cpu().numpy()
        mask_pred = mask_pred.cpu().numpy()

        pred_instances = []
        for i in range(cls_pred.shape[0]):
            pred = {}
            pred["scan_id"] = scan_ids[0]
            pred["label_id"] = cls_pred[i]
            pred["conf"] = score_pred[i]
            pred["pred_mask"] = rle_encode(mask_pred[i])
            pred_instances.append(pred)

        gt_instances = insts[0].gt_instances
        return dict(scan_id=scan_ids[0], pred_instances=pred_instances, gt_instances=gt_instances)

    def dbscan_post_processing(self, coords_float, mask_pred):
        curr_coords = coords_float.cpu().numpy()
        mask_pred_np = mask_pred.cpu().numpy()
        for i in range(len(mask_pred_np)):
            inst_id = np.where(mask_pred_np[i] == 1)[0]
            inst_points = curr_coords[inst_id]
            if inst_points.shape[0] != 0:
                clusters = DBSCAN(eps=self.dbscan_eps, min_samples=1, n_jobs=-1).fit(inst_points).labels_
                if -1 in np.unique(clusters):
                    print(f"Found noisy at mask {i}")
                new_mask = torch.zeros(mask_pred[i].shape, dtype=int)
                new_mask[inst_id] = torch.from_numpy(clusters) + 1
                new_mask = new_mask.cuda()
                for cluster_id in np.unique(clusters):
                    original_mask = mask_pred[i]
                    if cluster_id != -1:
                        mask_pred[i] = original_mask * (new_mask == cluster_id +1)
        return mask_pred

    def extract_feat(self, x, superpoints_h, superpoints_l, p2v_map):

        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        x = x.features[p2v_map.long()]
        x_h = scatter_mean(x, superpoints_h, dim=0)
        x_l = scatter_mean(x, superpoints_l, dim=0)
        return x_h, x_l
