'''
Author: wcl
Date: 2024-08-26 01:48:36
LastEditTime: 2024-09-27 20:33:30
Description: 
'''
from segment_anything import SamPredictor, build_sam, build_sam_hq, SamAutomaticMaskGenerator

class SAM_HQ:
    def __init__(self, cfg):
        # Segment Anything
        self.sam_predictor = SamPredictor(build_sam_hq(checkpoint=cfg.foundation_model.sam_checkpoint).to("cuda"))
        self.sam_generator = SamAutomaticMaskGenerator(build_sam_hq(checkpoint=cfg.foundation_model.sam_checkpoint).to("cuda"))
        print('------- Loaded Segment Anything HQ -------')