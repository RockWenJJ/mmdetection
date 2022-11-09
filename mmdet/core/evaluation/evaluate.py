import mmcv
import torch
import os.path as osp
from collections import OrderedDict
from mmcv.image import tensor2imgs
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def single_gpu_uie_evaluate(model,
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
            if k not in results.keys():
                results.setdefault(k, 0.)
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


def single_gpu_uie_det_evaluate(model,
                                data_loader,
                                show=False,
                                out_dir=None,
                                show_score_thr=0.3):
    model.eval()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    det_results = []
    uie_results = OrderedDict(ssim=0.,
                          psnr=0.,
                          loss_mse=0.,
                          loss_ssim=0.)
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            det_result, losses, images_dict = model(return_loss=False, **data)
        
        # parse det result
        batch_size = len(det_result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)
    
            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]
        
                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))
        
                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None
        
                model.module.show_result(
                    img_show,
                    result[i],
                    bbox_color=PALETTE,
                    text_color=PALETTE,
                    mask_color=PALETTE,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        det_results.extend(det_result)
        
        # parse uie result
        for k, v in losses.items():
            if k not in uie_results.keys():
                uie_results.setdefault(k, 0.)
            uie_results[k] += v.detach().cpu().numpy()
        
        img_metas = data['img_metas'].data[0] if not isinstance(data['img_metas'], list) else data['img_metas'][0].data[
            0]
        pred_tensor = images_dict['predict']
        targ_tensor = images_dict['target']
        
        pred_img = tensor2imgs(torch.clip(pred_tensor, 0, 1), **img_metas[0]['img_norm_cfg'])[0]
        targ_img = tensor2imgs(torch.clip(targ_tensor, 0, 1), **img_metas[0]['img_norm_cfg'])[0]
        
        uie_results['ssim'] += ssim(pred_img, targ_img, multichannel=True)
        uie_results['psnr'] += psnr(pred_img, targ_img)
        
        prog_bar.update()
    
    for k, v in uie_results.items():
        uie_results[k] /= len(data_loader)
    
    return det_results, uie_results