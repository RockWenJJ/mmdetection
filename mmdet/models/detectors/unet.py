import torch
import torch.nn as nn
from collections import OrderedDict
from mmcv.runner import BaseModule
from ..builder import build_backbone, build_neck, build_head
from ..builder import DETECTORS

@DETECTORS.register_module()
class UNet(BaseModule):
    CLASSES = None
    def __init__(self,
                 backbone,
                 neck,
                 head,
                 multi_scales=True,
                 **kwargs):
        '''
        :param backbone:
        :param neck:
        :param head:
        :param multi_scales: whether output features of different scales
        '''
        super(UNet, self).__init__()
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.head = build_head(head)
        self.multi_scales = multi_scales
    
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0]), {}
        
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)
    
    def forward_train(self, img, img_metas, **kwargs):
        assert 'input' in kwargs
        if self.multi_scales:
            assert isinstance(img, (list, tuple)), "when multi_scale, "
        input_img = kwargs['input'][-1] if self.multi_scales else kwargs['input']
        
        xs = self.forward_img(input_img)
        
        losses = dict()
        images_dict = dict()
        images_dict['input'] = input_img
        
        if self.multi_scales:
            assert len(xs) == len(img)
            images_dict['predict'] = xs[-1]
            images_dict['target'] = img[-1]
            for x, im in zip(xs, img):
                layer_loss = self.head.loss(x, im, img_metas)
                for k, v in layer_loss.items():
                    losses[k] = losses[k] + v if k in losses else v
            # losses.update(self.head.loss(x, img, img_metas))
        else:
            images_dict['predict'] = xs[-1]
            images_dict['target'] = img
            losses.update(self.head.loss(xs[-1], img, img_metas))
        
        return losses, images_dict
    
    def forward_test(self, img, img_metas=None, **kwargs):
        # input_img = kwargs['input']
        input_img = img
        xs = self.forward_img(input_img)
        # result = dict()
        # result['predict'] = x
        return xs[-1]
    
    def forward_dummy(self, img):
        '''Used for computing network flops and convert to onnx models'''
        xs = self.forward_img(img)
        return xs[-1]
    
    def train_step(self, data, optimizer):
        losses, images_dict = self(**data)
        loss, log_vars = self._parse_losses(losses)
        
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']), images=images_dict)
        
        return outputs
    
    def forward_img(self, img):
        xs = self.backbone(img)
        xs.reverse()
        xs = self.neck(xs)
        outs = []
        if self.multi_scales:
            for i, x in enumerate(xs):
                outs.append(self.head(x))
        else:
            outs.append(self.head(xs[-1]))
        
        return tuple(outs)
    
    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')
        
        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)
        
        # # If the loss_vars has different length, GPUs will wait infinitely
        # if dist.is_available() and dist.is_initialized():
        #     log_var_length = torch.tensor(len(log_vars), device=loss.device)
        #     dist.all_reduce(log_var_length)
        #     message = (f'rank {dist.get_rank()}' +
        #                f' len(log_vars): {len(log_vars)}' + ' keys: ' +
        #                ','.join(log_vars.keys()))
        #     assert log_var_length == len(log_vars) * dist.get_world_size(), \
        #         'loss log variables are different across GPUs!\n' + message
        
        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            # if dist.is_available() and dist.is_initialized():
            #     loss_value = loss_value.data.clone()
            #     dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()
        
        return loss, log_vars
    
    def onnx_export(self, img, img_metas):
        raise NotImplementedError(f'{self.__class__.__name__} does '
                                  f'not support ONNX EXPORT')