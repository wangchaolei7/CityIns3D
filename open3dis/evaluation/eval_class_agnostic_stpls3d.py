import os

import numpy as np
import torch
from segmenter3d.ISBNet.isbnet.util.rle import rle_decode
from open3dis.dataset.scannet200 import INSTANCE_CAT_SCANNET_200
from open3dis.dataset.scannetpp import SEMANTIC_CAT_SCANNET_PP, INSTANCE_BENCHMARK84_SCANNET_PP, SEMANTIC_INSTANCE_CAT_SCANNET_PP, INSTANCE_CAT_SCANNET_PP
from scannetv2_inst_eval import ScanNetEval
from tqdm import tqdm
import argparse
import yaml
from munch import Munch

from matterport3d_inst_eval import mattport3dEval
from open3dis.dataset.matterport3d import INSTANCE_CAT_Matterport3d

from stpls3d_inst_eval import stpls3dEval
from open3dis.dataset_outdoor.stpls3d import INSTANCE_CAT_STPLS3D
from plyfile import PlyData


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

    return parser


if __name__ == "__main__":

    args = get_parser().parse_args()
    cfg = Munch.fromDict(yaml.safe_load(open(args.config, "r").read()))

    eval_type = args.type

    if cfg.data.dataset_name == 'stpls3d':
        scan_eval = stpls3dEval(class_labels=INSTANCE_CAT_STPLS3D, use_label=False, dataset_name='stpls3d')
        pcl_path = cfg.data.gt_pth
        if eval_type == '2D':
            data_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.clustering_3d_output)
        if eval_type == '3D':
            data_path = os.path.join(cfg.data.cls_agnostic_3d_proposals_path)
        if eval_type == '2D_3D':
            pass

    scenes = sorted([s for s in os.listdir(data_path) if s.endswith(".pth")])
    gtsem = []
    gtinst = []
    res = []

    # VALID_SEMANTIC_IDS = list(range(1, 13))
    VALID_SEMANTIC_IDS = [1,2,3,4,5,6,7,8,9,10,11,12]  # STPLS3D 12类前景 类别2是否存在意味着是否去掉植被

    for scene in tqdm(scenes):
        scene_path = os.path.join(data_path, scene)
        try:
            pred_mask = torch.load(scene_path)
        except:
            print('SKIP: ', scene)
            continue

        gt_path = os.path.join(pcl_path, scene.replace(".pth", ".ply"))
        plydata = PlyData.read(gt_path)
        # import pdb; pdb.set_trace()
        vertex = plydata['vertex']
        sem_gt = vertex['semantic'].astype(np.int32)
        inst_gt = vertex['instance'].astype(np.int32)

        # import pdb; pdb.set_trace()
        valid_mask = np.isin(sem_gt, VALID_SEMANTIC_IDS)
        valid_mask = np.logical_and(valid_mask, inst_gt >= scan_eval.encode_value)

        sem_gt_filtered = sem_gt[valid_mask]
        inst_gt_filtered = inst_gt[valid_mask]

        if len(sem_gt_filtered) == 0:
            print(f'SKIP: {scene} has no valid foreground points')
            continue

        gtsem.append(sem_gt_filtered)
        gtinst.append(inst_gt_filtered)

        masks = pred_mask['ins']
        n_mask = len(masks)
        tmp = []
        for ind in range(n_mask):
            if isinstance(masks[ind], dict):
                mask = rle_decode(masks[ind])
            else:
                try:
                    mask = (masks[ind] == 1).numpy().astype(np.uint8)
                except:
                    mask = (masks[ind] == 1).astype(np.uint8)

            mask_filtered = mask[valid_mask]
            if np.count_nonzero(mask_filtered) == 0:
                continue

            conf = 1.0
            scene_id = scene.replace('.pth', '')
            tmp.append({"scan_id": scene_id, "label_id": 0, "conf": conf, "pred_mask": mask_filtered})
        res.append(tmp)

    scan_eval.evaluate(res, gtsem, gtinst)
