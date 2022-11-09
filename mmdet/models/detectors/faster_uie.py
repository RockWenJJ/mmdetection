# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from ..builder import build_neck, build_head


@DETECTORS.register_module()
class FasterUIE(TwoStageDetector):
    """A detector for both underwater object detection and image enhancement.
    """

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 uie_neck=None,
                 uie_head=None,
                 uie_multi_scales=False,
                 init_cfg=None):
        '''
        uie_neck: neck for underwater image enhancement
        uie_head: head for underwater image enhancement
        '''
        super(FasterUIE, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        
        if uie_neck is not None:
            self.uie_neck = build_neck(uie_neck)
        
        if uie_head is not None:
            self.uie_head = build_head(uie_head)
        
        self.uie_multi_scales = uie_multi_scales
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
    
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        # extract feat for object detection
        x = self.backbone(img)
        xs = x
        if self.with_neck: # fpn neck for detection
            x = self.neck(x)
        
        # for UIE
        xs = self.forward_uie(xs)
        
        losses = dict()
        images_dict = dict()
        images_dict['input'] = img

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)
        
        # Update losses for UIE
        assert 'cl_img' in kwargs, "for UIE_Det task, 'cl_img' must be specified"
        cl_target = kwargs['cl_img']
        if self.uie_multi_scales:
            assert len(xs) == len(img)
            images_dict['predict'] = xs[-1]
            images_dict['target'] = cl_target[-1]
            for x, im in zip(xs, cl_target):
                layer_loss = self.uie_head.loss(x, im, img_metas)
                for k, v in layer_loss.items():
                    losses[k] = losses[k] + v if k in losses else v
        else:
            images_dict['predict'] = xs[-1]
            images_dict['target'] = cl_target
            losses.update(self.uie_head.loss(xs[-1], cl_target, img_metas))
        
        return losses, images_dict
    
    def simple_test(self, img, img_metas=None, proposals=None, rescale=False, cl_img=None):
        # detection results
        det_results = super().simple_test(img, img_metas, proposals, rescale)
        
        # uie results
        losses = dict()
        images_dict = dict()
        images_dict['input'] = img
        
        xs = self.backbone(img)
        xs = self.forward_uie(xs)
        if self.uie_multi_scales:
            assert len(xs) == len(img)
            images_dict['predict'] = xs[-1]
            images_dict['target'] = cl_img[-1]
            for x, im in zip(xs, cl_img):
                layer_loss = self.uie_head.loss(x, im, img_metas)
                for k, v in layer_loss.items():
                    losses[k] = losses[k] + v if k in losses else v
        else:
            images_dict['predict'] = xs[-1]
            images_dict['target'] = cl_img[-1]
            losses.update(self.uie_head.loss(xs[-1], cl_img[-1], img_metas))
        
        return det_results, losses, images_dict

    def forward_uie(self, xs):
        xs.reverse()
        xs = self.uie_neck(xs)
        outs = []
        if self.uie_multi_scales:
            for i, x in enumerate(xs):
                outs.append(self.uie_head(x))
        else:
            outs.append(self.uie_head(xs[-1]))
        
        return tuple(outs)
    
    def train_step(self, data, optimizer):
        losses, images_dict = self(**data)
        loss, log_vars = self._parse_losses(losses)
        
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']), images=images_dict)
        
        return outputs
        
