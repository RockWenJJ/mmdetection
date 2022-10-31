import argparse
import os
import os.path as osp
import time

import mmcv
from mmcv import Config
import torch
import torch.distributed as dist
from mmdet.models import build_detector
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.core import *
from mmdet.utils import (get_device, build_dp)
from mmcv.runner import build_runner, get_dist_info, load_checkpoint
from mmcv.utils import get_logger
from mmcv.image import tensor2imgs
# from mmdet.apis import single_gpu_uie_test


def single_gpu_uie_test(model,
                        data_loader,
                        out_dir=None):
    model.eval()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)
        
        batch_size = 1
        if out_dir:
            # img_tensor = data['img'][0]
            img_tensor = result
            img_metas = data['img_metas'].data[0]
            # img = np.clip(img_tensor.squeeze().detach().cpu().numpy()*255, 0, 255).transpose(1, 2, 0)
            # imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR)]
            imgs = tensor2imgs(torch.clip(img_tensor, 0, 1), **img_metas[0]['img_norm_cfg'])
            
            for _, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]
                
                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))
                
                mmcv.imwrite(img_show, osp.join(out_dir, img_meta['filename']), auto_mkdir=True)
        
        for _ in range(batch_size):
            prog_bar.update()

def parse_args():
    parser = argparse.ArgumentParser(description='Test a UIE model')
    parser.add_argument('config', help='training config file path')
    parser.add_argument('--checkpoint', type=str, help='checkpoint file to load from')
    # parser.add_argument('--save-to', type=str, default=None, help='Directory to save the results')
    parser.add_argument('--out-dir', type=str, help='the dir to save results')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed training)')
    
    args = parser.parse_args()
    return args


def main():
    # 0. parse arguments
    args = parse_args()
    cfg = Config.fromfile(args.config)
    
    cfg.device = get_device()
    
    mmcv.mkdir_or_exist(osp.abspath(args.out_dir))
    
    test_dataloader_default_args = dict(samples_per_gpu=1,
                                        workers_per_gpu=0,
                                        shuffle=False)
    
    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }
    
    # build the dataloader
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    
    # build the model and load checkpoint
    model = build_detector(cfg.model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cuda:%d'%args.gpu_id)
    model = build_dp(model, cfg.device, device_ids=[args.gpu_id])
    
    # run test
    single_gpu_uie_test(model, data_loader, args.out_dir)


if __name__ == '__main__':
    main()