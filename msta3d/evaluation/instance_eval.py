# Referred to https://github.com/VinAIResearch/ISBNet/ and https://github.com/oneformer3d/oneformer3d
# Modified by Duc Tran

import numpy as np
from tqdm import tqdm
from copy import deepcopy
import multiprocessing as mp

from ..utils import rle_decode
from .instance_eval_util import get_instances
from msta3d.dataset.scannet200 import ScanNet200Dataset

class ScanNetEval(object):

    def __init__(self,
                 class_labels,
                 min_npoint=None,
                 iou_type=None,
                 use_label=True,
                 dataset_name="scannetv2"):

        self.dataset_name = dataset_name
        self.valid_class_labels = class_labels
        self.valid_class_ids = np.arange(len(class_labels)) + 1

        self.id2label = {}
        self.label2id = {}
        for i in range(len(self.valid_class_ids)):
            self.label2id[self.valid_class_labels[i]] = self.valid_class_ids[i]
            self.id2label[self.valid_class_ids[i]] = self.valid_class_labels[i]

        self.ious = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        if min_npoint:
            self.min_region_sizes = np.array([min_npoint])
        elif self.dataset_name == "stpls3d":
            self.min_region_sizes = np.array([10])
        else:
            self.min_region_sizes = np.array([100])
        self.distance_threshes = np.array([float('inf')])
        self.distance_confs = np.array([-float('inf')])

        self.iou_type = iou_type
        self.use_label = use_label
        if self.use_label:
            self.eval_class_labels = self.valid_class_labels
        else:
            self.eval_class_labels = ['class_agnostic']

    def evaluate_matches(self, matches):

        ious = self.ious
        min_region_sizes = [self.min_region_sizes[0]]
        dist_threshes = [self.distance_threshes[0]]
        dist_confs = [self.distance_confs[0]]

        ap = np.zeros((len(dist_threshes), len(self.eval_class_labels), len(ious)), float)
        pr_rc = np.zeros((2, len(self.eval_class_labels), len(ious)), float)
        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(zip(min_region_sizes, dist_threshes, dist_confs)):
            progress_bar = tqdm(total=len(ious))
            for oi, iou_th in enumerate(ious):
                pred_visited = {}
                for m in matches:
                    for p in matches[m]['pred']:
                        for label_name in self.eval_class_labels:
                            for p in matches[m]['pred'][label_name]:
                                if 'filename' in p:
                                    pred_visited[p['filename']] = False
                for li, label_name in enumerate(self.eval_class_labels):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False
                    for m in matches:
                        pred_instances = matches[m]['pred'][label_name]
                        gt_instances = matches[m]['gt'][label_name]
                        gt_instances = [
                            gt for gt in gt_instances
                            if gt['instance_id'] >= 1000 and gt['vert_count'] >= min_region_size
                            and gt['med_dist'] <= distance_thresh and gt['dist_conf'] >= distance_conf
                        ]
                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true = np.ones(len(gt_instances))
                        cur_score = np.ones(len(gt_instances)) * (-float('inf'))
                        cur_match = np.zeros(len(gt_instances), dtype=bool)

                        for (gti, gt) in enumerate(gt_instances):
                            found_match = False
                            for pred in gt['matched_pred']:
                                if pred_visited[pred['filename']]:
                                    continue
                                iou = pred['iou']
                                if iou > iou_th:
                                    confidence = pred['confidence']
                                    if cur_match[gti]:
                                        max_score = max(cur_score[gti], confidence)
                                        min_score = min(cur_score[gti], confidence)
                                        cur_score[gti] = max_score
                                        cur_true = np.append(cur_true, 0)
                                        cur_score = np.append(cur_score, min_score)
                                        cur_match = np.append(cur_match, True)
                                    else:
                                        found_match = True
                                        cur_match[gti] = True
                                        cur_score[gti] = confidence
                                        pred_visited[pred['filename']] = True
                            if not found_match:
                                hard_false_negatives += 1
                        cur_true = cur_true[cur_match == True]
                        cur_score = cur_score[cur_match == True]

                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred['matched_gt']:
                                iou = gt['iou']
                                if iou > iou_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                num_ignore = pred['void_intersection']
                                for gt in pred['matched_gt']:
                                    if gt['instance_id'] < 1000:
                                        num_ignore += gt['intersection']
                                    if (gt['vert_count'] < min_region_size or gt['med_dist'] > distance_thresh
                                            or gt['dist_conf'] < distance_conf):
                                        num_ignore += gt['intersection']
                                proportion_ignore = float(num_ignore) / pred['vert_count']
                                if proportion_ignore <= iou_th:
                                    cur_true = np.append(cur_true, 0)
                                    confidence = pred['confidence']
                                    cur_score = np.append(cur_score, confidence)

                        y_true = np.append(y_true, cur_true)
                        y_score = np.append(y_score, cur_score)

                    if has_gt and has_pred:
                        score_arg_sort = np.argsort(y_score)
                        y_score_sorted = y_score[score_arg_sort]
                        y_true_sorted = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        if len(y_true_sorted) == 0:
                            continue

                        (_, unique_indices) = np.unique(y_score_sorted, return_index=True)
                        num_prec_recall = len(unique_indices) + 1

                        num_examples = len(y_score_sorted)
                        num_true_examples = y_true_sorted_cumsum[-1]
                        precision = np.zeros(num_prec_recall)
                        recall = np.zeros(num_prec_recall)

                        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                        for idx_res, idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores - 1]
                            tp = num_true_examples - cumsum
                            fp = num_examples - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            p = float(tp) / (tp + fp)
                            r = float(tp) / (tp + fn)
                            precision[idx_res] = p
                            recall[idx_res] = r

                        precision[-1] = 1.
                        recall[-1] = 0.

                        f1_score = 2 * precision * recall / (precision + recall + 0.0001)
                        f1_argmax = f1_score.argmax()
                        best_pr = precision[f1_argmax]
                        best_rc = recall[f1_argmax]

                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.)

                        stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], 'valid')
                        ap_current = np.dot(precision, stepWidths)

                    elif has_gt:
                        ap_current = 0.0
                        best_pr = 0
                        best_rc = 0
                    else:
                        ap_current = float('nan')
                        best_pr = float('nan')
                        best_rc = float('nan')
                    ap[di, li, oi] = ap_current
                    pr_rc[0, li, oi] = best_pr
                    pr_rc[1, li, oi] = best_rc

                progress_bar.update()
            progress_bar.close()

        return ap, pr_rc

    def compute_averages(self, aps, pr_rc):

        d_pre = 0
        d_rec = 1
        o25 = np.where(np.isclose(self.ious, 0.25))
        o50 = np.where(np.isclose(self.ious, 0.5))
        oAllBut25 = np.where(np.logical_not(np.isclose(self.ious, 0.25)))

        avg_dict = {}
        avg_dict['all_ap'] = np.nanmean(aps[d_pre, :, oAllBut25])
        avg_dict['all_ap_25%'] = np.nanmean(aps[d_pre, :, o25])
        avg_dict['all_ap_50%'] = np.nanmean(aps[d_pre, :, o50])
        avg_dict['all_pc'] = np.nanmean(pr_rc[d_pre, :, oAllBut25])
        avg_dict['all_pc_25%'] = np.nanmean(pr_rc[d_pre, :, o25])
        avg_dict['all_pc_50%'] = np.nanmean(pr_rc[d_pre, :, o50])
        avg_dict['all_rc'] = np.nanmean(pr_rc[d_rec, :, oAllBut25])
        avg_dict['all_rc_25%'] = np.nanmean(pr_rc[d_rec, :, o25])
        avg_dict['all_rc_50%'] = np.nanmean(pr_rc[d_rec, :, o50])

        avg_dict['classes'] = {}
        for (li, label_name) in enumerate(self.eval_class_labels):
            avg_dict['classes'][label_name] = {}
            avg_dict['classes'][label_name]['ap'] = np.average(aps[d_pre, li, oAllBut25])
            avg_dict['classes'][label_name]['ap25%'] = np.average(aps[d_pre, li, o25])
            avg_dict['classes'][label_name]['ap50%'] = np.average(aps[d_pre, li, o50])
            avg_dict['classes'][label_name]['pc'] = np.average(pr_rc[d_pre, li, oAllBut25])
            avg_dict['classes'][label_name]['pc25%'] = np.average(pr_rc[d_pre, li, o25])
            avg_dict['classes'][label_name]['pc50%'] = np.average(pr_rc[d_pre, li, o50])
            avg_dict['classes'][label_name]['rc'] = np.average(pr_rc[d_rec, li, oAllBut25])
            avg_dict['classes'][label_name]['rc25%'] = np.average(pr_rc[d_rec, li, o25])
            avg_dict['classes'][label_name]['rc50%'] = np.average(pr_rc[d_rec, li, o50])
        return avg_dict

    def assign_instances_for_scan(self, preds, gts):

        gt_instances = get_instances(gts, self.valid_class_ids, self.valid_class_labels, self.id2label)
        if self.use_label:
            gt2pred = deepcopy(gt_instances)
            for label in gt2pred:
                for gt in gt2pred[label]:
                    gt['matched_pred'] = []

        else:
            gt2pred = {}
            agnostic_instances = []
            for _, instances in gt_instances.items():
                agnostic_instances += deepcopy(instances)
            for gt in agnostic_instances:
                gt['matched_pred'] = []
            gt2pred[self.eval_class_labels[0]] = agnostic_instances

        pred2gt = {}
        for label in self.eval_class_labels:
            pred2gt[label] = []
        num_pred_instances = 0
        bool_void = np.logical_not(np.in1d(gts // 1000, self.valid_class_ids))

        for pred in preds:
            if self.use_label:
                label_id = pred['label_id']
                if label_id not in self.id2label:
                    continue
                label_name = self.id2label[label_id]
            else:
                label_name = self.eval_class_labels[0]
            conf = pred['conf']
            pred_mask = pred['pred_mask']

            if isinstance(pred_mask, dict):
                pred_mask = rle_decode(pred_mask)
            assert pred_mask.shape[0] == gts.shape[0]

            pred_mask = np.not_equal(pred_mask, 0)
            num = np.count_nonzero(pred_mask)
            if num < self.min_region_sizes[0]:
                continue

            pred_instance = {}
            pred_instance['filename'] = '{}_{}'.format(pred['scan_id'], num_pred_instances)
            pred_instance['pred_id'] = num_pred_instances
            pred_instance['label_id'] = label_id if self.use_label else None
            pred_instance['vert_count'] = num
            pred_instance['confidence'] = conf
            pred_instance['void_intersection'] = np.count_nonzero(np.logical_and(bool_void, pred_mask))

            matched_gt = []
            for (gt_num, gt_inst) in enumerate(gt2pred[label_name]):
                intersection = np.count_nonzero(np.logical_and(gts == gt_inst['instance_id'], pred_mask))
                if intersection > 0:
                    gt_copy = gt_inst.copy()
                    pred_copy = pred_instance.copy()
                    gt_copy['intersection'] = intersection
                    pred_copy['intersection'] = intersection
                    iou = (float(intersection) / (gt_copy['vert_count'] + pred_copy['vert_count'] - intersection))
                    gt_copy['iou'] = iou
                    pred_copy['iou'] = iou
                    matched_gt.append(gt_copy)
                    gt2pred[label_name][gt_num]['matched_pred'].append(pred_copy)
            pred_instance['matched_gt'] = matched_gt
            num_pred_instances += 1
            pred2gt[label_name].append(pred_instance)

        return gt2pred, pred2gt

    def assign_boxes_for_scan(self, preds, gts, coord_floats):

        gt_instances = get_instances(gts, self.valid_class_ids, self.valid_class_labels, self.id2label, coord_floats=coord_floats)

        if self.use_label:
            gt2pred = deepcopy(gt_instances)
            for label in gt2pred:
                for gt in gt2pred[label]:
                    gt["matched_pred"] = []
        else:
            gt2pred = {}
            agnostic_instances = []
            for _, instances in gt_instances.items():
                agnostic_instances += deepcopy(instances)
            for gt in agnostic_instances:
                gt["matched_pred"] = []
            gt2pred[self.eval_class_labels[0]] = agnostic_instances

        pred2gt = {}
        for label in self.eval_class_labels:
            pred2gt[label] = []
        num_pred_instances = 0
        bool_void = np.logical_not(np.in1d(gts // 1000, self.valid_class_ids))

        for pred in preds:
            if self.use_label:
                label_id = pred["label_id"]
                if label_id not in self.id2label:
                    continue
                label_name = self.id2label[label_id]
            else:
                label_name = self.eval_class_labels[0]
            conf = pred["conf"]
            pred_mask = pred['pred_mask']
            if isinstance(pred_mask, dict):
                pred_mask = rle_decode(pred_mask)
            assert pred_mask.shape[0] == gts.shape[0]
            pred_mask = np.not_equal(pred_mask, 0)
            num = np.count_nonzero(pred_mask)
            if num < self.min_region_sizes[0]:
                continue

            pred_instance = {}
            pred_instance["filename"] = "{}_{}".format(pred["scan_id"], num_pred_instances)
            pred_instance["pred_id"] = num_pred_instances
            pred_instance["label_id"] = label_id if self.use_label else None
            pred_instance["confidence"] = conf
            pred_instance['vert_count'] = num
            pred_instance['void_intersection'] = np.count_nonzero(np.logical_and(bool_void, pred_mask))

            # pred_box_min = pred["pred_box"][:3]
            # pred_box_max = pred["pred_box"][3:]
            pred_box_min = coord_floats[pred_mask, :].min(0)
            pred_box_max = coord_floats[pred_mask, :].max(0)

            pred_vol = np.prod(np.clip((pred_box_max - pred_box_min), a_min=0.0, a_max=None), -1)

            matched_gt = []
            for (gt_num, gt_inst) in enumerate(gt2pred[label_name]):
                gt_box_min = gt_inst["box"][:3]
                gt_box_max = gt_inst["box"][3:]
                intersection = np.prod(
                    np.clip(
                        np.minimum(gt_box_max, pred_box_max) - np.maximum(gt_box_min, pred_box_min),
                        a_min=0.0,
                        a_max=None
                    ), -1
                )
                if intersection > 0:
                    gt_copy = gt_inst.copy()
                    pred_copy = pred_instance.copy()
                    gt_copy["intersection"] = intersection
                    pred_copy["intersection"] = intersection
                    gt_vol = np.prod(np.clip((gt_box_max - gt_box_min), a_min=0.0, a_max=None), -1)
                    iou = float(intersection) / (gt_vol + pred_vol - intersection)
                    gt_copy["iou"] = iou
                    pred_copy["iou"] = iou
                    matched_gt.append(gt_copy)
                    gt2pred[label_name][gt_num]["matched_pred"].append(pred_copy)

            pred_instance["matched_gt"] = matched_gt
            num_pred_instances += 1
            num_pred_instances += 1
            pred2gt[label_name].append(pred_instance)
            pred2gt[label_name].append(pred_instance)
        return gt2pred, pred2gt

    def print_results(self, avgs):

        sep = ''
        col1 = ':'
        lineLen = 90

        print()
        print('#' * lineLen)
        line = ''
        line += '{:<15}'.format('instance') + sep + col1
        line += '{:>8}'.format('AP') + sep
        line += '{:>8}'.format('AP_25%') + sep
        line += '{:>8}'.format('AP_50%') + sep
        line += '{:>8}'.format('PC') + sep
        line += '{:>8}'.format('PC_25%') + sep
        line += '{:>8}'.format('PC_50%') + sep
        line += '{:>8}'.format('RC') + sep
        line += '{:>8}'.format('RC_25%') + sep
        line += '{:>8}'.format('RC_50%') + sep

        print(line)
        print('#' * lineLen)

        for (_, label_name) in enumerate(self.eval_class_labels):
            ap_avg = avgs['classes'][label_name]['ap']
            ap_25o = avgs['classes'][label_name]['ap25%']
            ap_50o = avgs['classes'][label_name]['ap50%']
            pc_avg = avgs['classes'][label_name]['pc']
            pc_25o = avgs['classes'][label_name]['pc25%']
            pc_50o = avgs['classes'][label_name]['pc50%']
            rc_avg = avgs['classes'][label_name]['rc']
            rc_25o = avgs['classes'][label_name]['rc25%']
            rc_50o = avgs['classes'][label_name]['rc50%']
            line = '{:<15}'.format(label_name) + sep + col1
            line += sep + '{:>8.3f}'.format(ap_avg) + sep
            line += sep + '{:>8.3f}'.format(ap_25o) + sep
            line += sep + '{:>8.3f}'.format(ap_50o) + sep
            line += sep + '{:>8.3f}'.format(pc_avg) + sep
            line += sep + '{:>8.3f}'.format(pc_25o) + sep
            line += sep + '{:>8.3f}'.format(pc_50o) + sep
            line += sep + '{:>8.3f}'.format(rc_avg) + sep
            line += sep + '{:>8.3f}'.format(rc_25o) + sep
            line += sep + '{:>8.3f}'.format(rc_50o) + sep
            print(line)

        all_ap_avg = avgs['all_ap']
        all_ap_25o = avgs['all_ap_25%']
        all_ap_50o = avgs['all_ap_50%']
        all_pc_avg = avgs['all_pc']
        all_pc_25o = avgs['all_pc_25%']
        all_pc_50o = avgs['all_pc_50%']
        all_rc_avg = avgs['all_pc']
        all_rc_25o = avgs['all_rc_25%']
        all_rc_50o = avgs['all_rc_50%']

        print('-' * lineLen)
        line = '{:<15}'.format('average') + sep + col1
        line += '{:>8.3f}'.format(all_ap_avg) + sep
        line += '{:>8.3f}'.format(all_ap_25o) + sep
        line += '{:>8.3f}'.format(all_ap_50o) + sep
        line += '{:>8.3f}'.format(all_pc_avg) + sep
        line += '{:>8.3f}'.format(all_pc_25o) + sep
        line += '{:>8.3f}'.format(all_pc_50o) + sep
        line += '{:>8.3f}'.format(all_rc_avg) + sep
        line += '{:>8.3f}'.format(all_rc_25o) + sep
        line += '{:>8.3f}'.format(all_rc_50o) + sep
        print(line)
        print('#' * lineLen)
        print()

    def print_scannet200(self, avgs):

        print("ScanNet200 Evaluation")
        head_results, tail_results, common_results = [], [], []
        for (_, class_name) in enumerate(self.eval_class_labels):
            ap_avg = avgs["classes"][class_name]["ap"]
            ap_50o = avgs["classes"][class_name]["ap50%"]
            ap_25o = avgs["classes"][class_name]["ap25%"]

            if class_name not in ScanNet200Dataset.VALID_CLASS_IDS_200_VALIDATION:
                continue

            if class_name in ScanNet200Dataset.HEAD_CATS_SCANNET_200:
                head_results.append(np.array([ap_avg, ap_50o, ap_25o]))
            elif class_name in ScanNet200Dataset.COMMON_CATS_SCANNET_200:
                common_results.append(np.array([ap_avg, ap_50o, ap_25o]))
            elif class_name in ScanNet200Dataset.TAIL_CATS_SCANNET_200:
                tail_results.append(np.array([ap_avg, ap_50o, ap_25o]))
            else:
                raise ValueError("Unknown class name!!!")

        head_results = np.stack(head_results)
        common_results = np.stack(common_results)
        tail_results = np.stack(tail_results)

        mean_tail_results = np.nanmean(tail_results, axis=0)
        mean_common_results = np.nanmean(common_results, axis=0)
        mean_head_results = np.nanmean(head_results, axis=0)
        overall_ap_results = np.nanmean(np.vstack((head_results, common_results, tail_results)), axis=0)

        sep = ""
        col1 = ":"
        lineLen = 48

        print("#" * lineLen)
        line = ""
        line += "{:<15}".format("what") + sep + col1
        line += "{:>8}".format("AP") + sep
        line += "{:>8}".format("AP_50%") + sep
        line += "{:>8}".format("AP_25%") + sep

        print(line)
        print("#" * lineLen)
        line = "{:<15}".format("Head AP") + sep + col1
        line += "{:>8.3f}".format(mean_head_results[0]) + sep
        line += "{:>8.3f}".format(mean_head_results[1]) + sep
        line += "{:>8.3f}".format(mean_head_results[2]) + sep
        print(line)
        line = "{:<15}".format("Common AP") + sep + col1
        line += "{:>8.3f}".format(mean_common_results[0]) + sep
        line += "{:>8.3f}".format(mean_common_results[1]) + sep
        line += "{:>8.3f}".format(mean_common_results[2]) + sep
        print(line)
        line = "{:<15}".format("Tail AP") + sep + col1
        line += "{:>8.3f}".format(mean_tail_results[0]) + sep
        line += "{:>8.3f}".format(mean_tail_results[1]) + sep
        line += "{:>8.3f}".format(mean_tail_results[2]) + sep
        print(line)
        print("-" * lineLen)
        line = "{:<15}".format("AP") + sep + col1
        line += "{:>8.3f}".format(overall_ap_results[0]) + sep
        line += "{:>8.3f}".format(overall_ap_results[1]) + sep
        line += "{:>8.3f}".format(overall_ap_results[2]) + sep
        print(line)
        print("#" * lineLen)
        print()

    def write_result_file(self, avgs, filename):

        _SPLITTER = ','
        with open(filename, 'w') as f:
            f.write(_SPLITTER.join(['class', 'class id', 'ap', 'ap25', 'ap50', 'pc', 'pc25', 'pc50', 'rc', 'rc25', 'rc50']) + '\n')
            for class_name in self.eval_class_labels:
                ap = avgs['classes'][class_name]['ap']
                ap25 = avgs['classes'][class_name]['ap25%']
                ap50 = avgs['classes'][class_name]['ap50%']
                pc = avgs['classes'][class_name]['pc']
                pc25 = avgs['classes'][class_name]['pc25%']
                pc50 = avgs['classes'][class_name]['pc50%']
                rc = avgs['classes'][class_name]['rc']
                rc25 = avgs['classes'][class_name]['rc25%']
                rc50 = avgs['classes'][class_name]['rc50%']
                f.write(_SPLITTER.join([str(x) for x in [class_name, ap, ap25, ap50, pc, pc25, pc50, rc, rc25, rc50]]) + '\n')

    def evaluate(self, pred_list, gt_list, coord_floats):
        """
        Args:
            pred_list:
                for each scan:
                    for each instance
                        instance = dict(scan_id, label_id, mask, conf)
            gt_list:
                for each scan:
                    for each point:
                        gt_id = class_id * 1000 + instance_id
        """

        pool = mp.Pool()
        results = pool.starmap(self.assign_instances_for_scan, zip(pred_list, gt_list))
        pool.close()
        pool.join()

        matches = {}
        for i, (gt2pred, pred2gt) in enumerate(results):
            matches_key = f'gt_{i}'
            matches[matches_key] = {}
            matches[matches_key]['gt'] = gt2pred
            matches[matches_key]['pred'] = pred2gt
        print("Evaluate instance segmentation")
        ap_scores, pr_rc_scores = self.evaluate_matches(matches)
        avgs = self.compute_averages(ap_scores, pr_rc_scores)
        self.print_results(avgs)

        if self.dataset_name == "scannet200":
            self.print_scannet200(avgs)
        else:
            pool = mp.Pool()
            results_box = pool.starmap(self.assign_boxes_for_scan, zip(pred_list, gt_list, coord_floats))
            pool.close()
            pool.join()
            matches_box = {}
            for i, (gt2pred, pred2gt) in enumerate(results_box):
                matches_box_key = f'gt_{i}'
                matches_box[matches_box_key] = {}
                matches_box[matches_box_key]['gt'] = gt2pred
                matches_box[matches_box_key]['pred'] = pred2gt
            print("Evaluate object detection")
            ap_box_scores, pr_rc_box_scores = self.evaluate_matches(matches_box)
            avgs_box = self.compute_averages(ap_box_scores, pr_rc_box_scores)
            self.print_results(avgs_box)

            return avgs, avgs_box
        return avgs