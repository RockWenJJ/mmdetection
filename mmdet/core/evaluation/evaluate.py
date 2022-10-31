import mmcv
import torch
import os.path as osp
from collections import OrderedDict
from mmcv.image import tensor2imgs
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def single_gpu_evaluate(model,
                        data_loader,
                        out_dir=None):
    model.eval()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    
    results = OrderedDict(ssim=0.,
                          psnr=0.,
                          loss_mse=0.,
                          loss_ssim=0.)
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            losses, images_dict = model(return_loss=True, **data)
        
        for k, v in losses.items():
            results[k] += v.detach().cpu().numpy()
        
        img_metas = data['img_metas'].data[0] if not isinstance(data['img_metas'], list) else data['img_metas'][0].data[0]
        pred_tensor = images_dict['predict']
        targ_tensor = images_dict['target']
        
        pred_img = tensor2imgs(torch.clip(pred_tensor, 0, 1), **img_metas[0]['img_norm_cfg'])[0]
        targ_img = tensor2imgs(torch.clip(targ_tensor, 0, 1), **img_metas[0]['img_norm_cfg'])[0]
        
        results['ssim'] += ssim(pred_img, targ_img, multichannel=True)
        results['psnr'] += psnr(pred_img, targ_img)
        
        prog_bar.update()
    
    for k, v in results.items():
        results[k] /= len(data_loader)
    
    return results