import mmcv
import os
import numpy as np
from torch.utils.data import Dataset
import os.path as osp
from glob import glob

from .coco import CocoDataset
from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class UwCocoDataset(CocoDataset):
    CLASSES = ('holothurian', 'echinus', 'scallop', 'starfish')  # not useful for uie
    
    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 cl_prefix='_clear',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 file_client_args=dict(backend='disk'),
                 img_suffix='.png',
                 ):
        super().__init__(ann_file, pipeline,
                         classes=classes,
                         data_root=data_root,
                         img_prefix=img_prefix,
                         seg_prefix=seg_prefix,
                         proposal_file=proposal_file,
                         test_mode=test_mode,
                         filter_empty_gt=filter_empty_gt,
                         file_client_args=file_client_args)
        
        # self.img_prefix = os.path.join(img_prefix, 'image')
        self.img_suffix = img_suffix
        self.cl_prefix = self.img_prefix + cl_prefix if self.img_prefix[-1] != '/' \
            else self.img_prefix[:-1] + cl_prefix# clear image prefix
        
        
        self.data_infos = self.load_annotations(self.ann_file)
        self._set_group_flag()
        self.pipeline = Compose(pipeline)
    
    def __getitem__(self, idx):
        '''Get training/test data from pipeline.
        Args:
            idx (int): Index of data
        Returns:
            data (dict): Training/test data
        '''
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
    
    def prepare_train_img(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)
    
    def prepare_test_img(self, idx):
        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)
    
    def pre_pipeline(self, results):
        '''Prepare results dict for pipeline.'''
        results['img_prefix'] = self.img_prefix
        results['cl_prefix'] = self.cl_prefix

        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        
        return results
    
    def load_annotations(self, ann_file):
        '''Load annotation from annotation file.'''
        data_infos = super().load_annotations(ann_file)
        return data_infos
    
    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1
    
    def __len__(self):
        return len(self.data_infos)
    
    # def get_ann_info(self, idx):
    #     img_id = self.data_infos[idx]['id']
    #     ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
    #     ann_info = self.coco.load_anns(ann_ids)
    #     return self._parse_ann_info(self.data_infos[idx], ann_info)
    
    # def evaluate(self,
    #              results,
    #              metric='SSIM',
    #              logger=None):
    #     if not isinstance(metric, str):
    #         assert len(metric) == 1
    #         metric = metric[0]
    #
    #     eval_
    #     annotations = [self.get_ann_info(i) for i in range(len(self))]

