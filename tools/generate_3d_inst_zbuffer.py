import argparse
import json
import os
import time

# import matplotlib.pyplot as plt
import numpy as np
import clip
#### SigLip
# from transformers import AutoTokenizer, AutoProcessor, AutoModel
import torch
from open3dis.dataset_outdoor.stpls3d import INSTANCE_CAT_STPLS3D

# from open3dis.src.clustering.clustering_clone import process_hierarchical_agglomerative_spp, process_hierarchical_agglomerative_nospp, process_hierarchical_agglomerative_growspp
# from open3dis.src.clustering.clustering_zbuffer import process_hierarchical_agglomerative_growspp
# from open3dis.src.clustering.clustering_zbuffer_kitti import process_hierarchical_agglomerative_growspp_kitti
from open3dis.src.config_utils import load_yaml_config
from open3dis.src.clustering.clustering_zbuffer_stpls3d import process_hierarchical_agglomerative_growspp_stpls3d
from torch.nn import functional as F
from tqdm import tqdm, trange


def rle_encode_gpu_batch(masks):
    """
    Encode RLE (Run-length-encode) from 1D binary mask.
    Args:
        mask (np.ndarray): 1D binary mask
    Returns:
        rle (dict): encoded RLE
    """
    n_inst, length = masks.shape[:2]
    zeros_tensor = torch.zeros((n_inst, 1), dtype=torch.bool, device=masks.device)
    masks = torch.cat([zeros_tensor, masks, zeros_tensor], dim=1)

    rles = []
    for i in range(n_inst):
        mask = masks[i]
        runs = torch.nonzero(mask[1:] != mask[:-1]).view(-1) + 1

        runs[1::2] -= runs[::2]

        counts = runs.cpu().numpy()
        rle = dict(length=length, counts=counts)
        rles.append(rle)
    return rles


def rle_decode(rle):
    """
    Decode rle to get binary mask.
    Args:
        rle (dict): rle of encoded mask
    Returns:
        mask (np.ndarray): decoded mask
    """
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

# evaluate_openvocab = False
# evaluate_agnostic = False

class DeticMask:
    def __init__(self, pred_masks_rle=None, scores=None, pred_masks=None):
        self.pred_masks_rle = pred_masks_rle
        self.scores = scores
        self.pred_masks = pred_masks

def get_parser():
    parser = argparse.ArgumentParser(description="Configuration Open3DIS")
    parser.add_argument("--config",type=str,required = True,help="Config")
    parser.add_argument(
        "--config-overlay",
        action="append",
        default=[],
        help="Optional yaml overlay applied after --config; repeat this flag for multiple overlays.",
    )
    parser.add_argument(
        "--disable-splitting",
        action="store_true",
        help="Disable 3D splitting after lifting 2D masks; still run growing.",
    )
    return parser

if __name__ == "__main__":

    args = get_parser().parse_args()

    cfg = load_yaml_config(args.config, args.config_overlay)
    print(f"[3D clustering] config_stack={getattr(cfg, '_config_paths', [args.config])}")
    if args.disable_splitting:
        cfg.cluster.enable_splitting = False
        print("[3D clustering] splitting disabled: lifted 2D masks will skip clustering and go directly to growing.")

    evaluate_openvocab = cfg.evaluate.evalvocab  # Evaluation for openvocab
    evaluate_agnostic = cfg.evaluate.evalagnostic  # Evaluation for openvocab


    with open(cfg.data.split_path, "r") as file:
        scene_ids = sorted([line.rstrip("\n") for line in file])

    # Scannet200 class text features saving
    if cfg.data.dataset_name == 'scannet200':
        class_names = INSTANCE_CAT_SCANNET_200
    elif cfg.data.dataset_name == 'scannetpp':
        class_names = SEMANTIC_CAT_SCANNET_PP    
    elif cfg.data.dataset_name == 'kitti360':
        class_names = INSTANCE_CAT_KITTI360
    elif cfg.data.dataset_name == 'stpls3d':
        class_names = INSTANCE_CAT_STPLS3D
    else:
        raise ValueError(f"Unknown dataset: {cfg.data.dataset_name}")

    if evaluate_openvocab:
        scan_eval = ScanNetEval(class_labels=class_names, dataset_name=cfg.data.dataset_name)
        gtsem = []
        gtinst = []
        res = []

    ####CLIP    
    # clip_adapter, clip_preprocess = clip.load(cfg.foundation_model.clip_model, device = 'cuda')
    # with torch.no_grad(), torch.cuda.amp.autocast():
    #     text_features = clip_adapter.encode_text(clip.tokenize(class_names).cuda())
    #     text_features /= text_features.norm(dim=-1, keepdim=True)

    ####Siglip
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # siglip_model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(device)
    # siglip_tokenizer = AutoTokenizer.from_pretrained("google/siglip-so400m-patch14-384")
    # siglip_processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

    # text_inputs = [f"a photo of a {class_name}" for class_name in class_names]
    # inputs = siglip_tokenizer(text_inputs, padding="max_length", return_tensors="pt").to(device)
    # with torch.no_grad():
    #     text_features = siglip_model.get_text_features(**inputs)

    # Prepare directories
    save_dir_cluster = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.clustering_3d_output)
    os.makedirs(save_dir_cluster, exist_ok=True)
    save_dir_final = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.final_output) # final_output
    os.makedirs(save_dir_final, exist_ok=True)

    # Multiprocess logger
    if os.path.exists("track/tracker_lifted_stpls3d.txt") == False:
        with open("tracker_lifted__stpls3d.txt", "w") as file:
            file.write("Processed Scenes .\n")

    with torch.cuda.amp.autocast(enabled=cfg.fp16):
        start_time = time.time()
        for scene_id in tqdm(scene_ids):
            print("Process", scene_id)
            ## Tracker
            done = False
            path = scene_id + ".pth"
            with open("track/tracker_lifted_stpls3d.txt", "r") as file:
                lines = file.readlines()
                lines = [line.strip() for line in lines]
                for line in lines:
                    if path in line:
                        done = True
                        break
            if done == True:
                print("existed " + path)
                continue
            ## Write append each line
            # with open("tracker_lifted_zb_stpls3d.txt", "a") as file:
            #     file.write(path + "\n")

            #############################################
            # NOTE hierarchical agglomerative clustering
            if True:
                cluster_dict = None
                if cfg.final_instance.spp_level  == 'spp': # Group by Superpoints
                    proposals3d, confidence = process_hierarchical_agglomerative_spp(scene_id, cfg)
                elif cfg.final_instance.spp_level  == 'grow':
                    proposals3d, confidence = process_hierarchical_agglomerative_nospp(scene_id, cfg)                   
                elif cfg.final_instance.spp_level  == 'sppgrow':
                    proposals3d, confidence = process_hierarchical_agglomerative_growspp(scene_id, cfg)
                elif cfg.final_instance.spp_level  == 'sppgrow_kitti': # for kitti360
                    proposals3d, confidence = process_hierarchical_agglomerative_growspp_kitti(scene_id, cfg)
                elif cfg.final_instance.spp_level  == 'sppgrow_stpls3d': # for stpls3d
                    proposals3d, confidence = process_hierarchical_agglomerative_growspp_stpls3d(scene_id, cfg)
                else:
                    raise ValueError(f"Unknown spp level: {cfg.final_instance.spp_level}")
                if proposals3d == None: # Discarding too large scene
                    continue
                
                cluster_dict = {
                    "ins": rle_encode_gpu_batch(proposals3d),
                    "conf": confidence,
                }
                torch.save(cluster_dict, os.path.join(save_dir_cluster, f"{scene_id}.pth"))

            #############################################
            # NOTE get final instances
            if False:   
                cluster_dict = torch.load(os.path.join(save_dir_cluster, f"{scene_id}.pth"))
                masks_final, cls_final, scores_final = get_final_instances(
                    cfg,
                    text_features,
                    cluster_dict=cluster_dict,
                    use_2d_proposals=cfg.proposals.p2d,
                    use_3d_proposals=cfg.proposals.p3d,
                    only_instance=cfg.proposals.agnostic,
                )
                if scores_final == None:
                    final_dict = {
                        "ins": rle_encode_gpu_batch(masks_final),
                        "conf": None,
                        "final_class": None,
                    }
                else:
                    final_dict = {
                        "ins": rle_encode_gpu_batch(masks_final),
                        "conf": scores_final.cpu(),
                        "final_class": cls_final.cpu(),
                    }
                # NOTE Final instance
                torch.save(final_dict, os.path.join(save_dir_final, f"{scene_id}.pth"))
            #############################################
            # NOTE Evaluation openvocab
            # if evaluate_openvocab:
                
            #     if cfg.data.dataset_name == 's3dis':
            #         gt_path = os.path.join(cfg.data.gt_pth, f"{AREA}_{scene_id}.pth")
            #         _, _, sem_gt, inst_gt = torch.load(gt_path)
            #         n_points = len(sem_gt)
            #         if n_points > 1000000:
            #             stride = 8
            #         elif n_points >= 600000:
            #             stride = 6
            #         elif n_points >= 400000:
            #             stride = 2
            #         else:
            #             stride = 1
            #         sem_gt = sem_gt[::stride]
            #         inst_gt = inst_gt[::stride]

            #         # NOTE do not eval class "clutter"
            #         inst_gt[sem_gt==12] = -100
            #         sem_gt[sem_gt==12] = -100
            #     else:
            #         gt_path = os.path.join(cfg.data.gt_pth, f"{scene_id}.pth")
            #         _, _, sem_gt, inst_gt = torch.load(gt_path)

            #     gtsem.append(np.array(sem_gt).astype(np.int32))
            #     gtinst.append(np.array(inst_gt).astype(np.int32))

            #     masks_final = masks_final.cpu()
            #     cls_final = cls_final.cpu()

            #     n_mask = masks_final.shape[0]
            #     tmp = []
            #     for ind in range(n_mask):
            #         mask = (masks_final[ind] == 1).numpy().astype(np.uint8)
            #         conf = 1.0  # Same as OpenMask3D
            #         final_class = float(cls_final[ind])
            #         tmp.append({"scan_id": scene_id, "label_id": final_class + 1, "conf": conf, "pred_mask": mask})
            #     res.append(tmp)
            # NOTE Evaluation agnostic
            if evaluate_agnostic:
                pass

            print("Done")
            end_time = time.time()
            print(f"Time taken: {(end_time - start_time) / 60} mins")

        if evaluate_openvocab:
            scan_eval.evaluate(
                res, gtsem, gtinst, exp_path=os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.final_output)
            )
        if evaluate_agnostic:
            pass
