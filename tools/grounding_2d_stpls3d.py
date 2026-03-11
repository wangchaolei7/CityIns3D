import os
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm, trange
import cv2
import pickle

# Util
# from util2d.grounded_sam_original import Grounded_Sam
# from util2d.yoloworld_sam import YOLOWorld_SAM, YOLOWorld_SAM_Stpls3d
# from util2d.yoloworld_sam_nodepth import YOLOWorld_SAM # 没有深度图
from util2d.sam import Sam # only SAM
# from util2d.sam_rebuttal import Sam_Rebuttal
from util2d.sam_stpls3d import Sam_Stpls3d

from util2d.util import masks_to_rle

# from open3dis.dataset.scannet200 import INSTANCE_CAT_SCANNET_200 # Scannet200
from open3dis.dataset.kitti360 import INSTANCE_CAT_KITTI360 # 
from open3dis.dataset_outdoor.stpls3d import INSTANCE_CAT_STPLS3D

############################################## Foundations 2D + SAM ##############################################
'''
We generate class-agnostic 2D masks based on {DATASET} class name and CLIP features using 2D foundation models -> Speed
'''


def get_parser():
    parser = argparse.ArgumentParser(description="Configuration Open3DIS")
    parser.add_argument("--config",type=str,required = True,help="Config")
    return parser

if __name__ == "__main__":

    args = get_parser().parse_args()

    cfg = Munch.fromDict(yaml.safe_load(open(args.config, "r").read()))

    # Scannet split path
    with open(cfg.data.split_path, "r") as file:
        scene_ids = sorted([line.rstrip("\n") for line in file])

    if cfg.data.dataset_name == 'scannet200':
        class_names = INSTANCE_CAT_SCANNET_200
    elif cfg.data.dataset_name == 'kitti360':
        class_names = INSTANCE_CAT_KITTI360
    elif cfg.data.dataset_name == 'stpls3d':
        class_names = INSTANCE_CAT_STPLS3D
    else:
        raise ValueError(f"Unknown dataset: {cfg.data.dataset_name}")

    # Fondation model loader
    if cfg.segmenter2d.model == 'Grounded-SAM':
        model = Grounded_Sam(cfg)
    elif cfg.segmenter2d.model == 'YoloW-SAM':
        model = YOLOWorld_SAM(cfg)
    elif cfg.segmenter2d.model == 'YoloW-SAM_Stpls3d':  ### for Stpls3d
        model = YOLOWorld_SAM_Stpls3d(cfg)
    elif cfg.segmenter2d.model == 'SAM':
        model = Sam(cfg)
    elif cfg.segmenter2d.model == 'SAM_Rebuttal':  ### for rebuttal
        model = Sam_Rebuttal(cfg)
    elif cfg.segmenter2d.model == 'SAM_Stpls3d':  ### for Stpls3d
        model = Sam_Stpls3d(cfg)

    # Directory Init
    save_dir = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.mask2d_output)
    save_dir_feat = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.grounded_feat_output)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_feat, exist_ok=True)

    # Proces every scene
    with torch.cuda.amp.autocast(enabled=cfg.fp16):
        for scene_id in tqdm(scene_ids):
            # Tracker
            done = False
            path = scene_id + ".pth"
            with open("tracker_2d_stpls3d.txt", "r") as file:
                lines = file.readlines()
                lines = [line.strip() for line in lines]
                for line in lines:
                    if path in line:
                        done = True
                        break
            if done == True:
                print("existed " + path)
                continue
            # Write append each line
            with open("tracker_2d_stpls3d.txt", "a") as file:
                file.write(path + "\n")
            #####################################
            print("Process", scene_id)
            grounded_data_dict, grounded_features = model.gen_grounded_mask_and_feat(
                scene_id,
                class_names,
                cfg=cfg,
            )

            # continue

            # Save PC features
            torch.save({"feat": grounded_features}, os.path.join(save_dir_feat, scene_id + ".pth"))
            # Save 2D mask
            torch.save(grounded_data_dict, os.path.join(save_dir, scene_id + ".pth"))
            # Free memory
            torch.cuda.empty_cache()