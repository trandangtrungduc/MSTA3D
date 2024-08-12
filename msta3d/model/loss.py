import gorilla
import torch
import torch.nn as nn
import torch_scatter
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

@torch.jit.script
def batch_sigmoid_bce_loss(inputs: torch.Tensor, targets: torch.Tensor):

    N = inputs.shape[1]
    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction='none')
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction='none')
    loss = torch.einsum('nc,mc->nm', pos, targets) + torch.einsum('nc,mc->nm', neg, (1 - targets))

    return loss / N


@torch.jit.script
def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):

    inputs = inputs.sigmoid()
    numerator = 2 * torch.einsum('nc,mc->nm', inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)

    return loss

@torch.jit.script
def dice_loss(inputs: torch.Tensor, targets: torch.Tensor):

    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)

    return loss.mean()

def get_box_iou(inputs: torch.Tensor, targets: torch.Tensor):

    intersection = torch.prod(
        torch.clamp((torch.min(inputs[:, 3:], targets[:, 3:]) - torch.max(inputs[:, :3], targets[:, :3])), min=0.0),-1,)
    inputs_volumes = torch.prod(torch.clamp((inputs[:, 3:] - inputs[:, :3]), min=0.0), -1)
    targets_volumes = torch.prod(torch.clamp((targets[:, 3:] - targets[:, :3]), min=0.0), -1)
    union = inputs_volumes + targets_volumes - intersection
    iou = intersection / (union + 1e-6)

    return iou

def get_mask_iou(inputs: torch.Tensor, targets: torch.Tensor):

    inputs = inputs.sigmoid()
    binarized_inputs = (inputs >= 0.5).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)

    return score

class HungarianMatcher(nn.Module):

    def __init__(self, cost_weight):
        super().__init__()
        self.register_buffer('cost_weight', torch.tensor(cost_weight))

    @torch.no_grad()
    def forward(self,
                pred_labels,
                pred_masks,
                insts):

        indices = []
        for pred_label, pred_mask, inst in zip(pred_labels, pred_masks, insts):
            if len(inst) == 0:
                indices.append(([], []))
                continue
            pred_label = pred_label.softmax(-1)
            tgt_idx = inst.gt_labels
            cost_class = -pred_label[:, tgt_idx]

            tgt_mask = inst.gt_spmasks_h
            cost_mask = batch_sigmoid_bce_loss(pred_mask, tgt_mask.float())
            cost_dice = batch_dice_loss(pred_mask, tgt_mask.float())

            C = (self.cost_weight[0] * cost_class + self.cost_weight[1] * cost_mask + self.cost_weight[2] * cost_dice)
            C = C.cpu()

            indices.append(linear_sum_assignment(C))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


@gorilla.LOSSES.register_module()
class Criterion(nn.Module):

    def __init__(self,
                 ignore_label = -100,
                 loss_weight = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                 cost_weight = [1.0, 1.0, 1.0, 1.0],
                 non_object_weight = 0.1,
                 num_class = 18
                 ):

        super().__init__()
        class_weight = torch.ones(num_class + 1)
        class_weight[-1] = non_object_weight

        self.register_buffer('class_weight', class_weight)
        loss_weight = torch.tensor(loss_weight)
        self.register_buffer('loss_weight', loss_weight)
        self.matcher = HungarianMatcher(cost_weight)
        self.num_class = num_class
        self.ignore_label = ignore_label

    def _get_src_permutation_idx(self, indices):

        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_inst_info(self,
                      batched_gt_instance,
                      coords,
                      batch_offsets
                      ):

        for i, gt_inst in enumerate(batched_gt_instance):
            start_id = batch_offsets[i]
            end_id = batch_offsets[i + 1]
            coord = coords[start_id:end_id]
            inst_idx, point_idx = torch.nonzero(gt_inst['gt_masks'], as_tuple=True)
            inst_point = coord[point_idx]
            gt_inst['gt_center'] = torch_scatter.segment_coo(inst_point, inst_idx.cuda(), reduce='mean')

    def get_layer_loss(self,
                       layer,
                       aux_outputs,
                       insts
                       ):

        num_out = len(aux_outputs)
        if num_out > 4:
            pred_boxes = aux_outputs['boxes']
            pred_box_scores = aux_outputs['box_scores']
        pred_labels = aux_outputs['labels']
        pred_scores = aux_outputs['mask_scores']
        pred_masks = aux_outputs['masks']

        indices = self.matcher(pred_labels, pred_masks, insts)
        idx = self._get_src_permutation_idx(indices)

        loss_out = {}
        tgt_class_o = torch.cat([inst.gt_labels[idx_gt] for inst, (_, idx_gt) in zip(insts, indices)])
        tgt_class = torch.full(pred_labels.shape[:2],self.num_class,dtype=torch.int64,device=pred_labels.device)
        tgt_class[idx] = tgt_class_o
        class_loss = F.cross_entropy(pred_labels.transpose(1, 2), tgt_class, self.class_weight)
        loss_out['cls_loss'] = class_loss.item()

        score_loss = torch.tensor([0.0], device=pred_labels.device)
        mask_bce_loss = torch.tensor([0.0], device=pred_labels.device)
        mask_dice_loss = torch.tensor([0.0], device=pred_labels.device)

        if num_out > 4:
            box_l1_loss = torch.tensor([0.0], device=pred_labels.device)
            box_score_loss = torch.tensor([0.0], device=pred_labels.device)
            for box, box_score, inst, (idx_q, idx_gt) in zip(pred_boxes, pred_box_scores, insts, indices):
                if len(inst) == 0:
                    continue

                pred_box = box[idx_q]
                pred_box_score = box_score[idx_q]

                tgt_box = inst.gt_boxes[idx_gt]

                with torch.no_grad():
                    tgt_box_score = get_box_iou(pred_box, tgt_box).unsqueeze(1)

                filter_box_id = torch.where(tgt_box_score > 0.5)[0]
                if filter_box_id.numel():
                    tgt_box_score = tgt_box_score[filter_box_id]
                    pred_box_score = pred_box_score[filter_box_id]
                    box_score_loss += F.mse_loss(pred_box_score, tgt_box_score)

                box_l1_loss += F.l1_loss(pred_box, tgt_box)

            box_l1_loss /= len(pred_masks)
            box_score_loss /= len(pred_masks)

            loss_out['box_l1_loss'] = box_l1_loss.item()
            loss_out['box_score_loss'] = box_score_loss.item()

        for mask, score, inst, (idx_q, idx_gt) in zip(pred_masks, pred_scores, insts, indices):
            if len(inst) == 0:
                continue

            pred_mask = mask[idx_q]
            pred_score = score[idx_q]

            tgt_box = inst.gt_boxes[idx_gt]
            tgt_mask = inst.gt_spmasks_h[idx_gt]

            with torch.no_grad():
                tgt_score = get_mask_iou(pred_mask, tgt_mask).unsqueeze(1)

            filter_id = torch.where(tgt_score > 0.5)[0]
            if filter_id.numel():
                tgt_score = tgt_score[filter_id]
                pred_score = pred_score[filter_id]
                score_loss += F.mse_loss(pred_score, tgt_score)

            mask_dice_loss += dice_loss(pred_mask, tgt_mask.float())
            mask_bce_loss += F.binary_cross_entropy_with_logits(pred_mask, tgt_mask.float())

        score_loss /= len(pred_masks)
        mask_bce_loss /= len(pred_masks)
        mask_dice_loss /= len(pred_masks)

        loss_out['mask_score_loss'] = score_loss.item()
        loss_out['mask_bce_loss'] = mask_bce_loss.item()
        loss_out['mask_dice_loss'] = mask_dice_loss.item()

        if num_out > 4:
            loss = (
                self.loss_weight[0] * class_loss +
                self.loss_weight[1] * mask_bce_loss +
                self.loss_weight[2] * mask_dice_loss +
                self.loss_weight[3] * score_loss +
                self.loss_weight[4] * box_l1_loss +
                self.loss_weight[5] * box_score_loss
                )
        else:
            loss = (
                self.loss_weight[0] * class_loss +
                self.loss_weight[1] * mask_bce_loss +
                self.loss_weight[2] * mask_dice_loss +
                self.loss_weight[3] * score_loss
                )

        loss_out = {f'layer_{layer}_' + k: v for k, v in loss_out.items()}
        return loss, loss_out

    def forward(self, pred, insts):

        num_out = len(pred)
        if num_out > 4:
            pred_boxes = pred['boxes']
            pred_box_scores = pred['box_scores']
        pred_labels = pred['labels']
        pred_scores = pred['mask_scores']
        pred_masks = pred['masks']

        indices = self.matcher(pred_labels, pred_masks, insts)
        idx = self._get_src_permutation_idx(indices)

        loss_out = {}
        tgt_class_o = torch.cat([inst.gt_labels[idx_gt] for inst, (_, idx_gt) in zip(insts, indices)])
        tgt_class = torch.full(pred_labels.shape[:2], self.num_class, dtype=torch.int64,device=pred_labels.device,)
        tgt_class[idx] = tgt_class_o
        class_loss = F.cross_entropy(pred_labels.transpose(1, 2), tgt_class, self.class_weight)
        loss_out['cls_loss'] = class_loss.item()

        score_loss = torch.tensor([0.0], device=pred_labels.device)
        mask_bce_loss = torch.tensor([0.0], device=pred_labels.device)
        mask_dice_loss = torch.tensor([0.0], device=pred_labels.device)

        if num_out > 4:
            box_l1_loss = torch.tensor([0.0], device=pred_labels.device)
            box_score_loss = torch.tensor([0.0], device=pred_labels.device)
            for box, box_score, inst, (idx_q, idx_gt) in zip(pred_boxes, pred_box_scores, insts, indices):
                if len(inst) == 0:
                    continue

                pred_box = box[idx_q]
                pred_box_score = box_score[idx_q]

                tgt_box = inst.gt_boxes[idx_gt]

                with torch.no_grad():
                    tgt_box_score = get_box_iou(pred_box, tgt_box).unsqueeze(1)

                filter_box_id = torch.where(tgt_box_score > 0.5)[0]
                if filter_box_id.numel():
                    tgt_box_score = tgt_box_score[filter_box_id]
                    pred_box_score = pred_box_score[filter_box_id]
                    box_score_loss += F.mse_loss(pred_box_score, tgt_box_score)

                box_l1_loss += F.l1_loss(pred_box, tgt_box)

            box_l1_loss /= len(pred_masks)
            box_score_loss /= len(pred_masks)

            loss_out['box_l1_loss'] = box_l1_loss.item()
            loss_out['box_score_loss'] = box_score_loss.item()

        for mask, score, inst, (idx_q, idx_gt) in zip(pred_masks, pred_scores, insts, indices):
            if len(inst) == 0:
                continue

            pred_mask = mask[idx_q]
            pred_score = score[idx_q]

            tgt_mask = inst.gt_spmasks_h[idx_gt]

            with torch.no_grad():
                tgt_score = get_mask_iou(pred_mask, tgt_mask).unsqueeze(1)

            filter_id = torch.where(tgt_score > 0.5)[0]
            if filter_id.numel():
                tgt_score = tgt_score[filter_id]
                pred_score = pred_score[filter_id]
                score_loss += F.mse_loss(pred_score, tgt_score)

            mask_dice_loss += dice_loss(pred_mask, tgt_mask.float())
            mask_bce_loss += F.binary_cross_entropy_with_logits(pred_mask, tgt_mask.float())

        score_loss /= len(pred_masks)
        mask_bce_loss /= len(pred_masks)
        mask_dice_loss /= len(pred_masks)

        loss_out['mask_score_loss'] = score_loss.item()
        loss_out['mask_bce_loss'] = mask_bce_loss.item()
        loss_out['mask_dice_loss'] = mask_dice_loss.item()

        if num_out > 4:
            loss = (
                self.loss_weight[0] * class_loss +
                self.loss_weight[1] * mask_bce_loss +
                self.loss_weight[2] * mask_dice_loss +
                self.loss_weight[3] * score_loss +
                self.loss_weight[4] * box_l1_loss +
                self.loss_weight[5] * box_score_loss
                )
        else:
            loss = (
                self.loss_weight[0] * class_loss +
                self.loss_weight[1] * mask_bce_loss +
                self.loss_weight[2] * mask_dice_loss +
                self.loss_weight[3] * score_loss
                )

        if 'aux_outputs' in pred:
            for i, aux_outputs in enumerate(pred['aux_outputs']):
                loss_i, loss_out_i = self.get_layer_loss(i, aux_outputs, insts)
                loss += loss_i
                loss_out.update(loss_out_i)

        loss_out['loss'] = loss.item()

        return loss, loss_out