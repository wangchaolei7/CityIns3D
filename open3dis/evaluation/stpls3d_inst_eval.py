import multiprocessing as mp
import os
from copy import deepcopy

import numpy as np

from open3dis.evaluation.instance_eval_util import get_instances

from tqdm import tqdm, trange


class stpls3dEval(object):
    def __init__(self, class_labels, iou_type=None, use_label=True, dataset_name="stpls3d"):
        self.dataset_name = dataset_name
        self.encode_value = 1000
        self.valid_class_labels = class_labels
        self.valid_class_ids = np.arange(len(class_labels)) + 1
        self.id2label = {i + 1: lbl for i, lbl in enumerate(self.valid_class_labels)}
        self.label2id = {lbl: i + 1 for i, lbl in enumerate(self.valid_class_labels)}
        if self.dataset_name == "stpls3d":
            self.sem_label_mapping = {
                1: 1,   # Build -> building
                2: 2,   # Veg -> vegetation
                3: 3,   # Vehicle
                4: 4,   # Truck
                5: 5,   # Aircraft
                6: 6,   # Military vehicle
                7: 7,   # Bike
                8: 8,   # Motorcycle
                9: 9,   # LightPole
                10: 10, # StreetSign
                11: 11, # Clutter
                12: 12, # Fence
            }
        else:
            self.sem_label_mapping = {}
        self.ious = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        self.min_region_sizes = np.array([100])
        self.distance_threshes = np.array([float("inf")])
        self.distance_confs = np.array([-float("inf")])
        self.iou_type = iou_type
        self.use_label = use_label
        self.eval_class_labels = ["class_agnostic"] if not self.use_label else self.valid_class_labels

    def _build_encoded_gt_ids(self, gts_sem, gts_ins, gt_instance_format="auto"):
        gts_sem = np.asarray(gts_sem, dtype=np.int32)
        gts_ins = np.asarray(gts_ins, dtype=np.int32)

        if gt_instance_format == "auto":
            non_ignore = gts_ins >= 0
            if np.any(non_ignore):
                match_ratio = np.mean((gts_ins[non_ignore] // self.encode_value) == gts_sem[non_ignore])
                ge_encode_ratio = np.mean(gts_ins[non_ignore] >= self.encode_value)
                gt_instance_format = "encoded" if match_ratio > 0.95 and ge_encode_ratio > 0.95 else "raw"
            else:
                gt_instance_format = "raw"

        if gt_instance_format == "encoded":
            gt_sem_extracted = gts_ins // self.encode_value
            gt_ins_extracted = gts_ins % self.encode_value

            if not np.all(gt_sem_extracted == gts_sem):
                print("警告：gts_sem与gts_ins中的语义信息不一致，使用gts_ins中的语义")

            gts_sem_mapped = np.array([self.sem_label_mapping.get(label, 0) for label in gt_sem_extracted], dtype=np.int32)
            valid_mask = (gts_ins >= self.encode_value) & (gts_sem_mapped > 0)
            gts = np.zeros_like(gts_ins, dtype=np.int32)
            gts[valid_mask] = gts_sem_mapped[valid_mask] * self.encode_value + gt_ins_extracted[valid_mask]
            return gts, gt_instance_format

        if gt_instance_format != "raw":
            raise ValueError(f"Unsupported gt_instance_format: {gt_instance_format}")

        gts_sem_mapped = np.array([self.sem_label_mapping.get(label, 0) for label in gts_sem], dtype=np.int32)
        valid_mask = (gts_ins >= 0) & (gts_sem_mapped > 0)
        gts = np.zeros_like(gts_ins, dtype=np.int32)
        if not np.any(valid_mask):
            return gts, gt_instance_format

        for sem_id in np.unique(gts_sem_mapped[valid_mask]):
            sem_mask = valid_mask & (gts_sem_mapped == sem_id)
            raw_instance_ids = gts_ins[sem_mask]
            unique_raw_ids = np.unique(raw_instance_ids)
            if unique_raw_ids.shape[0] >= self.encode_value:
                raise ValueError(
                    f"Too many raw instances ({unique_raw_ids.shape[0]}) for semantic {sem_id}; "
                    f"cannot encode with base {self.encode_value}"
                )
            local_id_map = {int(raw_id): local_id for local_id, raw_id in enumerate(unique_raw_ids.tolist(), start=1)}
            local_ids = np.array([local_id_map[int(raw_id)] for raw_id in raw_instance_ids], dtype=np.int32)
            gts[sem_mask] = sem_id * self.encode_value + local_ids

        return gts, gt_instance_format

    def evaluate_matches(self, matches):
        ious = self.ious
        min_region_sizes = [self.min_region_sizes[0]]
        dist_threshes = [self.distance_threshes[0]]
        dist_confs = [self.distance_confs[0]]

        ap = np.zeros((len(dist_threshes), len(self.eval_class_labels), len(ious)), float)
        rc = np.zeros((len(dist_threshes), len(self.eval_class_labels), len(ious)), float)
        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
            zip(min_region_sizes, dist_threshes, dist_confs)
        ):
            for oi, iou_th in enumerate(ious):
                pred_visited = {}
                for m in matches:
                    for label_name in self.eval_class_labels:
                        for pred in matches[m]["pred"][label_name]:
                            if "filename" in pred:
                                pred_visited[pred["filename"]] = False
                for li, label_name in enumerate(self.eval_class_labels):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False
                    for m in matches:
                        pred_instances = matches[m]["pred"][label_name]
                        gt_instances = matches[m]["gt"][label_name]
                        gt_instances = [
                            gt
                            for gt in gt_instances
                            if gt["instance_id"] >= self.encode_value
                            and gt["vert_count"] >= min_region_size
                            and gt["med_dist"] <= distance_thresh
                            and gt["dist_conf"] >= distance_conf
                        ]
                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true = np.ones(len(gt_instances))
                        cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                        cur_match = np.zeros(len(gt_instances), dtype=bool)
                        for gti, gt in enumerate(gt_instances):
                            found_match = False
                            for pred in gt["matched_pred"]:
                                if pred_visited[pred["filename"]]:
                                    continue
                                iou = pred["iou"]
                                if iou > iou_th:
                                    confidence = pred["confidence"]
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
                                        pred_visited[pred["filename"]] = True
                            if not found_match:
                                hard_false_negatives += 1
                        cur_true = cur_true[cur_match == True]
                        cur_score = cur_score[cur_match == True]

                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred["matched_gt"]:
                                iou = gt["iou"]
                                if iou > iou_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                num_ignore = pred["void_intersection"]
                                for gt in pred["matched_gt"]:
                                    if gt["instance_id"] < self.encode_value:
                                        num_ignore += gt["intersection"]
                                    if (
                                        gt["vert_count"] < min_region_size
                                        or gt["med_dist"] > distance_thresh
                                        or gt["dist_conf"] < distance_conf
                                    ):
                                        num_ignore += gt["intersection"]
                                proportion_ignore = float(num_ignore) / pred["vert_count"]
                                if proportion_ignore <= iou_th:
                                    cur_true = np.append(cur_true, 0)
                                    confidence = pred["confidence"]
                                    cur_score = np.append(cur_score, confidence)

                        y_true = np.append(y_true, cur_true)
                        y_score = np.append(y_score, cur_score)

                    if has_gt and has_pred:
                        score_arg_sort = np.argsort(y_score)
                        y_score_sorted = y_score[score_arg_sort]
                        y_true_sorted = y_true[score_arg_sort]

                        if len(y_true_sorted) == 0:
                            ap_current = 0.0
                            rc_current = 0.0
                            continue

                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)
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

                        rc_current = recall[0]
                        precision[-1] = 1.0
                        recall[-1] = 0.0

                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.0)

                        stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], "valid")
                        ap_current = np.dot(precision, stepWidths)

                    elif has_gt:
                        ap_current = 0.0
                        rc_current = 0.0
                    else:
                        ap_current = float("nan")
                        rc_current = float("nan")
                    ap[di, li, oi] = ap_current
                    rc[di, li, oi] = rc_current
        return ap, rc

    def compute_averages(self, aps, rcs):
        d_inf = 0
        o50 = np.where(np.isclose(self.ious, 0.5))
        o25 = np.where(np.isclose(self.ious, 0.25))
        oAllBut25 = np.where(np.logical_not(np.isclose(self.ious, 0.25)))
        avg_dict = {}
        avg_dict["all_ap"] = np.nanmean(aps[d_inf, :, oAllBut25])
        avg_dict["all_ap_50%"] = np.nanmean(aps[d_inf, :, o50])
        avg_dict["all_ap_25%"] = np.nanmean(aps[d_inf, :, o25])
        avg_dict["all_rc"] = np.nanmean(rcs[d_inf, :, oAllBut25])
        avg_dict["all_rc_50%"] = np.nanmean(rcs[d_inf, :, o50])
        avg_dict["all_rc_25%"] = np.nanmean(rcs[d_inf, :, o25])
        avg_dict["classes"] = {}
        for li, label_name in enumerate(self.eval_class_labels):
            avg_dict["classes"][label_name] = {}
            avg_dict["classes"][label_name]["ap"] = np.average(aps[d_inf, li, oAllBut25])
            avg_dict["classes"][label_name]["ap50%"] = np.average(aps[d_inf, li, o50])
            avg_dict["classes"][label_name]["ap25%"] = np.average(aps[d_inf, li, o25])
            avg_dict["classes"][label_name]["rc"] = np.average(rcs[d_inf, li, oAllBut25])
            avg_dict["classes"][label_name]["rc50%"] = np.average(rcs[d_inf, li, o50])
            avg_dict["classes"][label_name]["rc25%"] = np.average(rcs[d_inf, li, o25])
        return avg_dict

    def assign_instances_for_scan(self, preds, gts_sem, gts_ins, gt_instance_format="auto"):
        gts, gt_instance_format = self._build_encoded_gt_ids(
            gts_sem,
            gts_ins,
            gt_instance_format=gt_instance_format,
        )

        gt_instances = get_instances(gts, self.valid_class_ids, self.valid_class_labels,
                                    self.id2label, dataset=self.dataset_name)

        gt2pred = {}
        agnostic_instances = []
        for _, instances in gt_instances.items():
            agnostic_instances += deepcopy(instances)
        for gt in agnostic_instances:
            gt["matched_pred"] = []
        gt2pred["class_agnostic"] = agnostic_instances
        gt_index_by_id = {
            int(gt["instance_id"]): gt_idx for gt_idx, gt in enumerate(gt2pred["class_agnostic"])
        }

        pred2gt = {"class_agnostic": []}
        num_pred_instances = 0

        for pred in preds:
            conf = pred["conf"]
            pred_mask = pred["pred_mask"]
            if pred_mask.shape[0] != gts.shape[0]:
                print(f"警告: 跳过无效预测 {pred['scan_id']}，预测长度{pred_mask.shape[0]}≠真实长度{gts.shape[0]}")
                continue

            pred_mask = np.not_equal(pred_mask, 0)
            num = np.count_nonzero(pred_mask)
            if num < self.min_region_sizes[0]:
                continue

            pred_instance = {
                "filename": f"{pred['scan_id']}_{num_pred_instances}",
                "pred_id": num_pred_instances,
                "label_id": None,
                "vert_count": num,
                "confidence": conf,
                "void_intersection": 0,
            }

            pred_gts = gts[pred_mask]
            pred_instance["void_intersection"] = int(np.count_nonzero(pred_gts == 0))

            matched_gt = []
            matched_gt_ids = pred_gts[pred_gts > 0]
            if matched_gt_ids.size > 0:
                unique_gt_ids, intersections = np.unique(matched_gt_ids, return_counts=True)
                for gt_instance_id, intersection in zip(unique_gt_ids.tolist(), intersections.tolist()):
                    gt_num = gt_index_by_id.get(int(gt_instance_id))
                    if gt_num is None:
                        continue
                    gt_inst = gt2pred["class_agnostic"][gt_num]
                    gt_copy = gt_inst.copy()
                    pred_copy = pred_instance.copy()
                    gt_copy["intersection"] = int(intersection)
                    pred_copy["intersection"] = int(intersection)
                    iou = float(intersection) / (gt_copy["vert_count"] + pred_copy["vert_count"] - intersection)
                    gt_copy["iou"] = iou
                    pred_copy["iou"] = iou
                    matched_gt.append(gt_copy)
                    gt2pred["class_agnostic"][gt_num]["matched_pred"].append(pred_copy)
            pred_instance["matched_gt"] = matched_gt
            num_pred_instances += 1
            pred2gt["class_agnostic"].append(pred_instance)

        return gt2pred, pred2gt

    def print_results(self, avgs):
        sep = ""
        col1 = ":"
        lineLen = 64

        print()
        print("#" * lineLen)
        line = ""
        line += "{:<15}".format("what") + sep + col1
        line += "{:>8}".format("AP") + sep
        line += "{:>8}".format("AP_50%") + sep
        line += "{:>8}".format("AP_25%") + sep
        line += "{:>8}".format("AR") + sep
        line += "{:>8}".format("RC_50%") + sep
        line += "{:>8}".format("RC_25%") + sep

        print(line)
        print("#" * lineLen)

        for li, label_name in enumerate(self.eval_class_labels):
            ap_avg = avgs["classes"][label_name]["ap"]
            ap_50o = avgs["classes"][label_name]["ap50%"]
            ap_25o = avgs["classes"][label_name]["ap25%"]
            rc_avg = avgs["classes"][label_name]["rc"]
            rc_50o = avgs["classes"][label_name]["rc50%"]
            rc_25o = avgs["classes"][label_name]["rc25%"]
            line = "{:<15}".format(label_name) + sep + col1
            line += sep + "{:>8.3f}".format(ap_avg) + sep
            line += sep + "{:>8.3f}".format(ap_50o) + sep
            line += sep + "{:>8.3f}".format(ap_25o) + sep
            line += sep + "{:>8.3f}".format(rc_avg) + sep
            line += sep + "{:>8.3f}".format(rc_50o) + sep
            line += sep + "{:>8.3f}".format(rc_25o) + sep
            print(line)

        all_ap_avg = avgs["all_ap"]
        all_ap_50o = avgs["all_ap_50%"]
        all_ap_25o = avgs["all_ap_25%"]
        all_rc_avg = avgs["all_rc"]
        all_rc_50o = avgs["all_rc_50%"]
        all_rc_25o = avgs["all_rc_25%"]

        print("-" * lineLen)
        line = "{:<15}".format("average") + sep + col1
        line += "{:>8.3f}".format(all_ap_avg) + sep
        line += "{:>8.3f}".format(all_ap_50o) + sep
        line += "{:>8.3f}".format(all_ap_25o) + sep
        line += "{:>8.3f}".format(all_rc_avg) + sep
        line += "{:>8.3f}".format(all_rc_50o) + sep
        line += "{:>8.3f}".format(all_rc_25o) + sep
        print(line)
        print("#" * lineLen)
        print()

    def write_result_file(self, avgs, filename):
        _SPLITTER = ","
        with open(filename, "w") as f:
            f.write(_SPLITTER.join(["class", "class id", "ap", "ap50", "ap25"]) + "\n")
            for class_name in self.eval_class_labels:
                ap = avgs["classes"][class_name]["ap"]
                ap50 = avgs["classes"][class_name]["ap50%"]
                ap25 = avgs["classes"][class_name]["ap25%"]
                f.write(_SPLITTER.join([str(x) for x in [class_name, ap, ap50, ap25]]) + "\n")

    def _scene_output_filename(self, output_tag=""):
        if self.use_label:
            filename = "scene_ovis_results.txt"
        else:
            filename = "zAPresults/stpls3d_zbuffer/txt_uto_img16_s12.txt"
        if not output_tag:
            return filename
        base, ext = os.path.splitext(filename)
        return f"{base}_{output_tag}{ext}"

    def _finalize_results(self, matches, scene_results, exp_path="./", output_tag=""):
        ap_scores, rc_scores = self.evaluate_matches(matches)
        avgs = self.compute_averages(ap_scores, rc_scores)

        scene_results = list(scene_results)
        scene_results.sort(key=lambda x: x["ap"], reverse=True)

        output_path = os.path.join(exp_path, self._scene_output_filename(output_tag))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            f.write(f"# all_ap,{avgs['all_ap']:.6f}\n")
            f.write(f"# all_ap_50,{avgs['all_ap_50%']:.6f}\n")
            f.write(f"# all_ap_25,{avgs['all_ap_25%']:.6f}\n")
            f.write(f"# all_rc,{avgs['all_rc']:.6f}\n")
            f.write(f"# all_rc_50,{avgs['all_rc_50%']:.6f}\n")
            f.write(f"# all_rc_25,{avgs['all_rc_25%']:.6f}\n")
            f.write("scene_id,AP,AP_50%,AP_25%\n")
            for result in scene_results:
                f.write(f"{result['scene_id']},{result['ap']:.3f},{result['ap50']:.3f},{result['ap25']:.3f}\n")

        result_filename = "result.txt" if not output_tag else f"result_{output_tag}.txt"
        self.write_result_file(avgs, os.path.join(exp_path, result_filename))
        self.print_results(avgs)

        return avgs

    def evaluate_precomputed(self, scene_entries, exp_path="./", output_tag=""):
        matches = {}
        scene_results = []
        for i, entry in enumerate(scene_entries):
            matches_key = f"gt_{i}"
            matches[matches_key] = {
                "gt": entry["gt"],
                "pred": entry["pred"],
            }
            scene_results.append(
                {
                    "scene_id": entry["scene_id"],
                    "ap": entry["ap"],
                    "ap50": entry["ap50"],
                    "ap25": entry["ap25"],
                }
            )
        return self._finalize_results(matches, scene_results, exp_path=exp_path, output_tag=output_tag)

    def evaluate(self, pred_list, gt_sem_list, gt_ins_list, exp_path="./", output_tag=""):
        scene_results = []

        results = []
        for i in trange(len(gt_sem_list)):
            results.append((self.assign_instances_for_scan(pred_list[i], gt_sem_list[i], gt_ins_list[i])))

        matches = {}
        for i, (gt2pred, pred2gt) in enumerate(results):
            matches_key = f"gt_{i}"
            matches[matches_key] = {}
            matches[matches_key]["gt"] = gt2pred
            matches[matches_key]["pred"] = pred2gt

            scene_matches = {matches_key: matches[matches_key]}
            scene_ap, scene_rc = self.evaluate_matches(scene_matches)
            scene_avgs = self.compute_averages(scene_ap, scene_rc)

            scene_name = pred_list[i][0]["scan_id"] if pred_list[i] else f"scene_{i}"
            scene_results.append({
                "scene_id": scene_name,
                "ap": scene_avgs["all_ap"],
                "ap50": scene_avgs["all_ap_50%"],
                "ap25": scene_avgs["all_ap_25%"]
            })

        return self._finalize_results(matches, scene_results, exp_path=exp_path, output_tag=output_tag)
