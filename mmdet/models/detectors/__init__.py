# Copyright (c) OpenMMLab. All rights reserved.
from .atss import ATSS
from .autoassign import AutoAssign
from .base import BaseDetector
from .cascade_rcnn import CascadeRCNN
from .centernet import CenterNet
from .cornernet import CornerNet
from .ddod import DDOD
from .deformable_detr import DeformableDETR
from .detr import DETR
from .fast_rcnn import FastRCNN
from .faster_rcnn import FasterRCNN
from .fcos import FCOS
from .fovea import FOVEA
from .fsaf import FSAF
from .gfl import GFL
from .grid_rcnn import GridRCNN
from .htc import HybridTaskCascade
from .kd_one_stage import KnowledgeDistillationSingleStageDetector
from .lad import LAD
from .mask2former import Mask2Former
from .mask_rcnn import MaskRCNN
from .mask_scoring_rcnn import MaskScoringRCNN
from .maskformer import MaskFormer
from .nasfcos import NASFCOS
from .paa import PAA
from .panoptic_fpn import PanopticFPN
from .panoptic_two_stage_segmentor import TwoStagePanopticSegmentor
from .point_rend import PointRend
from .queryinst import QueryInst
from .reppoints_detector import RepPointsDetector
from .retinanet import RetinaNet
from .rpn import RPN
from .scnet import SCNet
from .single_stage import SingleStageDetector
from .solo import SOLO
from .solov2 import SOLOv2
from .sparse_rcnn import SparseRCNN
from .tood import TOOD
from .trident_faster_rcnn import TridentFasterRCNN
from .two_stage import TwoStageDetector
from .vfnet import VFNet
from .yolact import YOLACT
from .yolo import YOLOV3
from .yolof import YOLOF
from .yolox import YOLOX
from .unet import UNet
from .faster_uie import FasterUIE
from .mae import MaskedAutoencoderViT
from .unet2 import UNet2
from .restormer import Restormer
from .restormer2 import Restormer2
from .restormer3 import Restormer3
from .uformer import Uformer
from .watr_v1 import WaTrV1
from .watr_v2 import WaTrV2
from .watr_v3 import WaTrV3
from .watr_v4 import WaTrV4
from .watr_v5 import WaTrV5
from .watr_v6 import WaTrV6
from .ushape_trans import UshapeTrans
from .udaformer import UDAformer
from .ursct import URSCT_SR
from .waterformer_v1 import WaterFormerV1
from .waterformer_v2 import WaterFormerV2
from .waterformer_v3 import WaterFormerV3
from .syreanet import SyreaNet
from .swinir import SwinIR

__all__ = [
    'ATSS', 'BaseDetector', 'SingleStageDetector', 'TwoStageDetector', 'RPN',
    'KnowledgeDistillationSingleStageDetector', 'FastRCNN', 'FasterRCNN',
    'MaskRCNN', 'CascadeRCNN', 'HybridTaskCascade', 'RetinaNet', 'FCOS',
    'GridRCNN', 'MaskScoringRCNN', 'RepPointsDetector', 'FOVEA', 'FSAF',
    'NASFCOS', 'PointRend', 'GFL', 'CornerNet', 'PAA', 'YOLOV3', 'YOLACT',
    'VFNet', 'DETR', 'TridentFasterRCNN', 'SparseRCNN', 'SCNet', 'SOLO',
    'SOLOv2', 'DeformableDETR', 'AutoAssign', 'YOLOF', 'CenterNet', 'YOLOX',
    'TwoStagePanopticSegmentor', 'PanopticFPN', 'QueryInst', 'LAD', 'TOOD',
    'MaskFormer', 'DDOD', 'Mask2Former', 'UNet', 'FasterUIE', 'MaskedAutoencoderViT',
    'UNet2', 'Restormer', 'Restormer2', 'Restormer3', 'Uformer', 'WaTrV1', 'WaTrV2',
    'WaTrV3', 'WaTrV4', 'WaTrV5', 'WaTrV6', 'UshapeTrans', 'UDAformer', 'URSCT_SR',
    'WaterFormerV1', 'WaterFormerV2', 'WaterFormerV3', 'SyreaNet', 'SwinIR'
]
