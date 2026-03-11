import torch
import numpy as np
import random
import os

import pyviz3d.visualizer as viz
from plyfile import PlyData
from os.path import join
import open3d as o3d

def generate_palette(n):
    palette = []
    for _ in range(n):
        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)
        palette.append((red, green, blue))
    return palette

def rle_decode(rle):
    length = rle["length"]
    s = rle["counts"]

    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask

def read_pointcloud(pcd_path):
    scene_pcd = o3d.io.read_point_cloud(str(pcd_path))
    point = np.asarray(scene_pcd.points)
    color = np.asarray(scene_pcd.colors)

    return point, color

SCANNET200 = 'wall.floor.cabinet.bed.chair.sofa.table.door.window.bookshelf.picture.counter.desk.curtrain.refridgerator.shower_toilet.sink.bathtub.otherfurniture'
class_names = SCANNET200.split('.')

class VisualizationScannet200:
    def __init__(self, point, color):
        self.point = point.astype(np.float32)
        self.color = color
        self.vis = viz.Visualizer()
        self._center_point_cloud()
        self.vis.add_points('pcl', self.point, self.color.astype(np.float32), point_size=20, visible=True)

    def _center_point_cloud(self):
        """让点云以几何中心为原点，便于浏览器初始视角对准十字中心。"""
        if self.point.size == 0:
            return
        centroid = self.point.mean(axis=0, keepdims=True)
        self.point = self.point - centroid
    
    def save(self, path):
        self.vis.save(path)
    
    def superpointviz(self, spp_path):
        print('...Visualizing Superpoints...')
        # spp = torch.from_numpy(torch.load(spp_path)).to(device='cuda')
        spp = torch.load(spp_path).to(device='cuda')
        unique_spp, spp, num_point = torch.unique(spp, return_inverse=True, return_counts=True)
        n_spp = unique_spp.shape[0]
        pallete =  generate_palette(n_spp + 1)
        uniqueness = torch.unique(spp).clone()
        # skip -1 
        tt_col = self.color.copy()
        for i in range(0, uniqueness.shape[0]):
            ss = torch.where(spp == uniqueness[i].item())[0]
            for ind in ss:
                tt_col[ind,:] = pallete[int(uniqueness[i].item())]
        self.vis.add_points(f'superpoint: ' + str(i), self.point, tt_col, point_size=20, visible=True)
        print('---Done---')
    
    def gtviz(self, gt_data, specific = False):
        print('...Visualizing Groundtruth...')
        if gt_data.endswith('.ply'):
            plydata = PlyData.read(gt_data)
            vertex = plydata['vertex']
            sem_label = np.asarray(vertex['semantic'], dtype=np.int32)
            ins_label = np.asarray(vertex['instance'], dtype=np.int32)
        else:
            gt_pack = torch.load(gt_data)
            if isinstance(gt_pack, (list, tuple)) and len(gt_pack) == 4:
                _, _, sem_label, ins_label = gt_pack
            else:
                raise ValueError(f"Unsupported GT format: {gt_data}")
            if isinstance(sem_label, torch.Tensor):
                sem_label = sem_label.cpu().numpy()
            if isinstance(ins_label, torch.Tensor):
                ins_label = ins_label.cpu().numpy()
        pallete =  generate_palette(int(2e3 + 1))
        n_label = np.unique(ins_label)
        tt_col = self.color.copy()
        for i in range(0, n_label.shape[0]):
            sem_value = sem_label[np.where(ins_label==n_label[i])][0]
            if sem_value in (0, 15, 17, 18, 19):  # 忽略地面
                continue
            tt_col[np.where(ins_label==n_label[i])] = pallete[i]
            if specific: # be more specific
                tt_col_specific = self.color.copy()
                tt_col_specific[np.where(ins_label==n_label[i])] = pallete[i]
                self.vis.add_points(f'GT instance: {i}_{sem_value}', self.point, tt_col_specific, point_size=20, visible=True)

        self.vis.add_points(f'GT instance: ' + str(i), self.point, tt_col, point_size=20, visible=True)
        print('---Done---')

    def vizmask3d(self, mask3d_path, specific = False):
        print('...Visualizing 3D backbone mask...')
        dic = torch.load(mask3d_path)
        instance = dic['ins']
        try:
            instance = torch.stack([torch.tensor(rle_decode(ins)) for ins in instance])
        except:
            pass
        conf3d = dic['conf']
        pallete =  generate_palette(int(2e3 + 1))
        tt_col = self.color.copy()
        limit = 10
        for i in range(0, instance.shape[0]):
            tt_col[instance[i] == 1] = pallete[i]
            if specific and limit > 0: # be more specific but limit 10 masks (avoiding lag)
                limit -= 1
                tt_col_specific = self.color.copy()
                tt_col_specific[instance[i] == 1] = pallete[i]
                self.vis.add_points(f'3D backbone mask: ' + str(i) + '_' + str(conf3d[i]), self.point, tt_col_specific, point_size=20, visible=True)

        self.vis.add_points(f'3D backbone mask: ' + str(i), self.point, tt_col, point_size=20, visible=True)
        print('---Done---')

    def vizmask2d(self, mask2d_path, specific = False):
        print('...Visualizing 2D lifted mask...')
        dic = torch.load(mask2d_path)
        instance = dic['ins']
        instance = torch.stack([torch.tensor(rle_decode(ins)) for ins in instance])
        conf2d = dic['conf'] # confidence really doesn't affect much (large mask -> small conf)
        pallete =  generate_palette(int(5e3 + 1))
        tt_col = self.color.copy()
        limit = 10
        for i in range(0, instance.shape[0]):
            tt_col[instance[i] == 1] = pallete[i]
            if specific and limit > 0: # be more specific but limit 10 masks (avoiding lag)
                limit -= 1
                tt_col_specific = self.color.copy()
                tt_col_specific[instance[i] == 1] = pallete[i]
                self.vis.add_points(f'2D lifted mask: ' + str(i) + '_' + str(conf2d[i].item())[:5], self.point, tt_col_specific, point_size=20, visible=True)

        self.vis.add_points(f'2D lifted mask: ' + str(i), self.point, tt_col, point_size=20, visible=True)
        print('---Done---')        
        
    def finalviz(self, agnostic_path, specific = False, vocab = False):
        print('...Visualizing final class agnostic mask...')
        dic = torch.load(agnostic_path)
        instance = dic['ins']
        instance = torch.stack([torch.tensor(rle_decode(ins)) for ins in instance])
        conf2d = dic['conf'] # confidence really doesn't affect much (large mask -> small conf)

        if vocab == True:
            label = dic['final_class']
        pallete =  generate_palette(int(2e3 + 1))
        tt_col = self.color.copy()
        limit = 5
        for i in range(0, instance.shape[0]):
            tt_col[instance[i] == 1] = pallete[i]
            if specific and limit > 0: # be more specific but limit 10 masks (avoiding lag)
                limit -= 1
                tt_col_specific = self.color.copy()
                tt_col_specific[instance[i] == 1] = pallete[i]
                if vocab == True:
                    self.vis.add_points(f'final mask: ' + str(i) + '_' + class_names[label[i]], self.point, tt_col_specific, point_size=20, visible=True)                
                else:
                    self.vis.add_points(f'final mask: ' + str(i) + '_' + str(conf2d[i].item())[:5], self.point, tt_col_specific, point_size=20, visible=True)

        self.vis.add_points(f'final mask: ' + str(i), self.point, tt_col, point_size=20, visible=True)
        print('---Done---')  

    def featureviz(self, feature_path):
        print('...Visualizing final class agnostic mask...')
        # breakpoint()
        dic = torch.load(feature_path)['feat']
        pallete =  generate_palette(int(2e3 + 1))
        tt_col = self.color.copy()
        feat = torch.mean(dic, dim = -1)
        feat = feat - torch.min(feat).item()
        feat*=1000
        breakpoint()
        feat = torch.nn.functional.normalize(feat, dim = -1)
        for i in range(self.point.shape[0]):
            tt_col = tt_col[i, :]*feat[i].item()

        self.vis.add_points(f'feature: _', self.point, tt_col, point_size=20, visible=True)                
        print('---Done---')  

if __name__ == "__main__":
    
    '''
        Visualization using PyViz3D
        1. superpoint visualization
        2. ground-truth annotation
        3. 3D backbone mask (isbnet, mask3d) -- class-agnostic
        4. lifted 2D masks -- class-agnostic
        5. final masks --class-agnostic (2D+3D)
        
    
    '''
    # Scene ID to visualize
    # scene_id = '10_points_GTv3_01'
    # scene_id = '20_points_GTv3_68'
    # scene_id = '25_points_GTv3_55'
    # scene_id = '05_points_GTv3_97'
    scene_id = '10_points_GTv3_42'
    # scene_id = '25_points_GTv3_99'

    # scene_id = '25_points_GTv3_09'


    ##### The format follows the dataset tree
    ## 1
    check_superpointviz = True
    spp_path = '/home/Data/data2/wcl/DataSet/STPLS3D_Open3DIS/3D/superpoints/' + scene_id + '.pt'
    ## 2
    check_gtviz = True
    gt_path = '/home/Data/data2/wcl/DataSet/STPLS3D_Open3DIS/3D/groundtruth/' + scene_id + '.ply'
    ## 3
    check_3dviz = False
    mask3d_path = './data/Scannet200/Scannet200_3D/val/isbnet_clsagnostic_scannet200/' + scene_id + '.pth'
    ## 4
    check_2dviz = True
    # mask2d_path = '/home/Data/data2/wcl/Open3DIS/exp_stpls3d/version_SAM/hier_agglo_spp_dbscan/' + scene_id + '.pth'
    mask2d_path = '/home/Data/data2/wcl/Open3DIS/exp_stpls3d/version_SAM/hier_agglo_spp_dbscan_test/' + scene_id + '.pth'
    ## 5
    check_finalviz = False
    agnostic_path = '../exp/version_detic/final_result_hier_agglo/' + scene_id + '.pth'
    # 6
    check_featureviz = False
    feature_path = '../exp/version_check/refined_grounded_feat/' + scene_id + '.pth'


    pyviz3d_dir = '../viz' # visualization directory
    # Visualize Point Cloud 
    ply_file = '/home/Data/data2/wcl/DataSet/STPLS3D_Open3DIS/3D/original_ply_files'
    point, color = read_pointcloud(os.path.join(ply_file,scene_id + '.ply'))
    color = color * 127.5

    VIZ = VisualizationScannet200(point, color)    
    
    if check_superpointviz:
        VIZ.superpointviz(spp_path)
    if check_gtviz:
        VIZ.gtviz(gt_path, specific = False)
    if check_3dviz:
        VIZ.vizmask3d(mask3d_path, specific = False)
    if check_2dviz:
        VIZ.vizmask2d(mask2d_path, specific = False)
    if check_finalviz:
        VIZ.finalviz(agnostic_path, specific = False, vocab = False)
    if check_featureviz:
        VIZ.featureviz(feature_path)
    VIZ.save(pyviz3d_dir)
