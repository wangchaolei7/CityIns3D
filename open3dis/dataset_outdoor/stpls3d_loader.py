### 2025.02.20

import cv2
import numpy as np
import os
import torch
import open3d as o3d

class STPLS3DReader(object):
    def __init__(
        self,
        root_path,
        cfg
    ):
        self.root_path = root_path
        self.scene_id = os.path.basename(root_path)


        depth_folder = os.path.join(self.root_path, "depth")
        if not os.path.exists(depth_folder):
            self.depth_frame_ids = []
        else:
            depth_images = os.listdir(depth_folder)
            self.depth_frame_ids = sorted([x.split(".")[0] for x in depth_images])

        print("Number of depth original frames:", len(self.depth_frame_ids))

        color_folder = os.path.join(self.root_path, "color")
        if not os.path.exists(color_folder):
            self.frame_ids = []
        else:
            color_images = os.listdir(color_folder)
            self.frame_ids = sorted([x.split(".")[0] for x in color_images])  

        print("Number of color original frames:", len(self.frame_ids))

        # self.global_intrinsic = np.array(
        #     [[552.554261, 0, 682.049453],
        #     [0, 552.554261, 238.769549],
        #     [0, 0, 1]]
        # ) # 没有global_intrinsic

        self.depth_scale = 1 # 保存为实际值
        
        intrinsic_file = os.path.join(self.root_path, "intrinsic.txt")
        try:
            self.intrinsic = np.loadtxt(intrinsic_file)
        except:
            self.intrinsic = None
            print('No global intrinsic')

        self.scene_pcd_path = os.path.join(cfg.data.original_ply, f"{self.scene_id}.ply")

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.frame_ids)

    def read_depth(self, depth_path):
        depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE) # 强制单通道读取
        depth_image = depth_image / self.depth_scale  # rescale to obtain depth in meters
        # h, w = depth_image.shape
        # nz_left = np.count_nonzero(depth_image[:, : w//2] > 0)
        # nz_right = np.count_nonzero(depth_image[:, w//2 :] > 0)
        # print("depth nonzero: left=", nz_left, " right=", nz_right)
        return depth_image

    def read_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def read_pose(self, pose_path):
        pose = np.load(pose_path)
        return pose
    
    def read_intrinsic(self, intrinsic_path):
        intrinsic = np.load(intrinsic_path)
        return intrinsic

    def read_valid_mask(self, valid_mask_path):
        if not valid_mask_path or not os.path.exists(valid_mask_path):
            return None
        valid_mask = cv2.imread(valid_mask_path, cv2.IMREAD_GRAYSCALE)
        if valid_mask is None:
            return None
        return valid_mask > 0

    def read_coverage(self, coverage_path):
        if not coverage_path or not os.path.exists(coverage_path):
            return None
        coverage = cv2.imread(coverage_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
        return coverage

    def read_frame_meta(self, meta_path):
        if not meta_path or not os.path.exists(meta_path):
            return {}
        with np.load(meta_path) as data:
            return {key: data[key] for key in data.files}

    def read_pointcloud(self, pcd_path=None):
        if pcd_path is None:
            pcd_path = self.scene_pcd_path
        scene_pcd = o3d.io.read_point_cloud(str(pcd_path))
        point = np.array(scene_pcd.points)

        return point
    
    # def read_gt_3D(self, gt_path):
    #     _, _, sem_gt, inst_gt = torch.load(gt_path)
    #     return sem_gt, inst_gt
    
    def read_spp(self, spp_path, device='cuda'):
        spp = torch.load(spp_path)
        if isinstance(spp, np.ndarray):
            spp = torch.from_numpy(spp)
        spp = spp.to(device)

        return spp
    
    def read_feature(self, feat_path, device='cuda'):
        dc_feature = torch.load(feat_path)
        if isinstance(dc_feature, dict) and "feat" in dc_feature:
            dc_feature = dc_feature["feat"]
        if isinstance(dc_feature, np.ndarray):
            dc_feature = torch.from_numpy(dc_feature)
        
        dc_feature = dc_feature.to(device)
        return dc_feature
    
    def read_3D_proposal(self, agnostic3d_path):
        agnostic3d_data = torch.load(agnostic3d_path)
        return agnostic3d_data

    def __getitem__(self, idx):
        """
        Returns:
            frame: a dict
                {frame_id}: str
                {depth}: (h, w)
                {image}: (h, w)
                {image_path}: str
                {intrinsics}: np.array 3x3
                {pose}: np.array 4x4
                {pcd}: np.array (n, 3)
                    in world coordinate
                {color}: (n, 3)
        """
        frame_id = self.frame_ids[idx]
        frame = {}
        frame["frame_id"] = frame_id
        framedepth = "{}.tif".format(frame_id) # 保存格式为tif
        framecolor = "{}.png".format(frame_id) # 保存格式为png
        framepose = "{}.npy".format(frame_id)
        frameintrinsic = "{}.npy".format(frame_id)

        depth_image_path = os.path.join(self.root_path, "depth", framedepth)
        image_path = os.path.join(self.root_path, "color", framecolor)
        pose_path = os.path.join(self.root_path, "pose", framepose)
        intrinsic_path = os.path.join(self.root_path, "intrinsic", frameintrinsic)
        valid_mask_path = os.path.join(self.root_path, "valid_mask", framecolor)
        coverage_path = os.path.join(self.root_path, "coverage", framecolor)
        meta_path = os.path.join(self.root_path, "meta", f"{frame_id}.npz")
        
        frame["depth_path"] = depth_image_path
        frame["image_path"] = image_path
        frame["pose_path"] = pose_path
        frame["intrinsic_path"] = intrinsic_path
        frame["valid_mask_path"] = valid_mask_path
        frame["coverage_path"] = coverage_path
        frame["meta_path"] = meta_path


        return frame
