import os
import multiprocessing as mp

import numpy as np
import torch
from tqdm import tqdm
import argparse
import yaml
from munch import Munch

from stpls3d_inst_eval import stpls3dEval
from open3dis.dataset_outdoor.stpls3d import INSTANCE_CAT_STPLS3D
from open3dis.dataset_outdoor.stpls3d_io import load_semantic_instance, resolve_scene_path


def torch_load_local(path):
    return torch.load(path, weights_only=False)


def rle_decode(rle):
    length = rle["length"]
    try:
        s = rle["counts"].split()
    except:
        s = rle["counts"]

    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask


def get_parser():
    parser = argparse.ArgumentParser(description="Configuration Open3DIS")
    parser.add_argument("--config", type=str, required=True, help="Config")
    parser.add_argument("--type", type=str, required=True, help="[2D, 3D, 2D_3D]")
    parser.add_argument("--num-workers", type=int, default=0, help="Parallel scene workers, 0 for auto")
    parser.add_argument(
        "--protocol",
        type=str,
        default="cityins3d",
        choices=["cityins3d", "sai3d"],
        help="Evaluation protocol: current CityIns3D filtered foreground or SAI3D-style full-scene ignore logic",
    )

    return parser


def process_scene_for_eval(task):
    (
        scene,
        data_path,
        pcl_path,
        valid_semantic_ids,
        class_labels,
        dataset_name,
        protocol,
    ) = task

    scene_path = os.path.join(data_path, scene)
    try:
        pred_mask = torch_load_local(scene_path)
    except Exception as exc:
        return {"status": "skip", "scene": scene, "reason": f"load failed: {exc}"}

    scene_id = scene.replace(".pth", "")
    gt_path = resolve_scene_path(pcl_path, scene_id)
    sem_gt, inst_gt = load_semantic_instance(gt_path)
    gt_ext = os.path.splitext(gt_path)[1].lower()
    gt_instance_format = "raw" if gt_ext == ".txt" else "encoded"

    scan_eval = stpls3dEval(class_labels=class_labels, use_label=False, dataset_name=dataset_name)
    valid_mask = np.isin(sem_gt, valid_semantic_ids)
    if gt_instance_format == "raw":
        valid_mask = np.logical_and(valid_mask, inst_gt >= 0)
    else:
        valid_mask = np.logical_and(valid_mask, inst_gt >= scan_eval.encode_value)
    if not np.any(valid_mask):
        return {"status": "skip", "scene": scene, "reason": "has no valid foreground points"}

    if protocol == "cityins3d":
        eval_sem = sem_gt[valid_mask]
        eval_inst = inst_gt[valid_mask]
    elif protocol == "sai3d":
        eval_sem = sem_gt
        eval_inst = inst_gt
    else:
        return {"status": "skip", "scene": scene, "reason": f"unknown protocol {protocol}"}

    masks = pred_mask['ins']
    confs = pred_mask.get("conf", None)
    tmp = []
    for ind, encoded_mask in enumerate(masks):
        if isinstance(encoded_mask, dict):
            mask = rle_decode(encoded_mask)
        else:
            try:
                mask = (encoded_mask == 1).numpy().astype(np.uint8)
            except Exception:
                mask = (encoded_mask == 1).astype(np.uint8)

        if protocol == "cityins3d":
            eval_mask = mask[valid_mask]
        else:
            eval_mask = mask

        if np.count_nonzero(eval_mask) == 0:
            continue

        conf = 1.0
        if confs is not None and len(confs) > ind:
            conf = float(confs[ind])
        tmp.append({"scan_id": scene_id, "label_id": 0, "conf": conf, "pred_mask": eval_mask})

    gt2pred, pred2gt = scan_eval.assign_instances_for_scan(
        tmp,
        eval_sem,
        eval_inst,
        gt_instance_format=gt_instance_format,
    )
    scene_matches = {"gt_0": {"gt": gt2pred, "pred": pred2gt}}
    scene_ap, scene_rc = scan_eval.evaluate_matches(scene_matches)
    scene_avgs = scan_eval.compute_averages(scene_ap, scene_rc)

    return {
        "status": "ok",
        "scene_id": scene_id,
        "gt": gt2pred,
        "pred": pred2gt,
        "ap": scene_avgs["all_ap"],
        "ap50": scene_avgs["all_ap_50%"],
        "ap25": scene_avgs["all_ap_25%"],
    }


if __name__ == "__main__":

    args = get_parser().parse_args()
    cfg = Munch.fromDict(yaml.safe_load(open(args.config, "r").read()))

    eval_type = args.type
    protocol = args.protocol

    if cfg.data.dataset_name == 'stpls3d':
        scan_eval = stpls3dEval(class_labels=INSTANCE_CAT_STPLS3D, use_label=False, dataset_name='stpls3d')
        pcl_path = cfg.data.gt_pth
        if eval_type == '2D':
            data_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.clustering_3d_output_lifted_part) # clustering_3d_output_lifted_part
        if eval_type == '3D':
            data_path = os.path.join(cfg.data.cls_agnostic_3d_proposals_path)
        if eval_type == '2D_3D':
            pass

    scenes = sorted([s for s in os.listdir(data_path) if s.endswith(".pth")])
    VALID_SEMANTIC_IDS = [1,2,3,4,5,6,7,8,9,10,11,12]  # STPLS3D 12类前景 类别2是否存在意味着是否去掉植被
    print(f"[eval] protocol={protocol} scenes={len(scenes)} data_path={data_path}")
    num_workers = args.num_workers
    if num_workers <= 0:
        num_workers = min(32, max(1, os.cpu_count() or 1))

    tasks = [
        (
            scene,
            data_path,
            pcl_path,
            VALID_SEMANTIC_IDS,
            INSTANCE_CAT_STPLS3D,
            cfg.data.dataset_name,
            protocol,
        )
        for scene in scenes
    ]

    scene_entries = []
    if num_workers == 1:
        iterator = map(process_scene_for_eval, tasks)
        for result in tqdm(iterator, total=len(tasks)):
            if result["status"] != "ok":
                print(f"SKIP: {result['scene']} {result['reason']}")
                continue
            scene_entries.append(result)
    else:
        start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        ctx = mp.get_context(start_method)
        with ctx.Pool(processes=num_workers) as pool:
            iterator = pool.imap_unordered(process_scene_for_eval, tasks, chunksize=1)
            for result in tqdm(iterator, total=len(tasks)):
                if result["status"] != "ok":
                    print(f"SKIP: {result['scene']} {result['reason']}")
                    continue
                scene_entries.append(result)

    scan_eval.evaluate_precomputed(scene_entries, exp_path="./", output_tag=protocol)
