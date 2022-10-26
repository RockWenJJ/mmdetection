import cv2
import mmcv
import numpy as np
import random
import json as js
import os.path as osp

from ..builder import PIPELINES


@PIPELINES.register_module()
class Synthesize:
    '''Synthesize underwater images from in-air image&depth.'''
    
    def __init__(self,
                 coef_path,
                 rand=False,
                 num=1,
                 multiply_depth=10,
                 add_depth=2):
        super().__init__()
        
        self.coefs = js.load(open(coef_path, 'r'))
        self.rand = rand
        self.num = num
        self.multiply_depth = multiply_depth
        self.add_depth = add_depth
    
    def _estimate_backscattering(self, depths, B_inf, beta_B, J_prime, beta_D_prime):
        val = (B_inf * (1 - np.exp(-1 * beta_B * depths))) + (J_prime * np.exp(-1 * beta_D_prime * depths))
        return val
    
    def _calculate_beta_D(self, depths, a, b, c, d):
        return (a * np.exp(b * depths)) + (c * np.exp(d * depths))
    
    def _scale(self, img):
        return (img - np.min(img)) / (np.max(img) - np.min(img))
    
    def _degrade_image(self, img, depth, B, beta_D, wbalance):
        img = img / np.expand_dims(wbalance, axis=0)
        img = self._scale(img)
        
        t = np.exp(-beta_D * np.expand_dims(depth, axis=2))
        
        degrade = img * t + B
        degrade = np.maximum(0.0, np.minimum(1., degrade))
        
        return degrade, t, B
    
    def _synthesize(self, img, dep, coef):
        # extracting coeffs
        Bcoefs_r, Bcoefs_g, Bcoefs_b = np.array(coef["Bcoefs_r"]), np.array(coef["Bcoefs_g"]), np.array(
            coef["Bcoefs_b"])
        Dcoefs_r, Dcoefs_g, Dcoefs_b = np.array(coef["Dcoefs_r"]), np.array(coef["Dcoefs_g"]), np.array(
            coef["Dcoefs_b"])
        wbalance = np.array(coef['wbalance'])
        
        # estimate backscattering
        Br = self._estimate_backscattering(dep, *Bcoefs_r)
        Bg = self._estimate_backscattering(dep, *Bcoefs_g)
        Bb = self._estimate_backscattering(dep, *Bcoefs_b)
        
        # estimate direct transmission
        beta_D_r = self._calculate_beta_D(dep, *Dcoefs_r) * 0.5  # TODO
        beta_D_g = self._calculate_beta_D(dep, *Dcoefs_g) * 0.5
        beta_D_b = self._calculate_beta_D(dep, *Dcoefs_b) * 0.5
        
        B = np.stack([Br, Bg, Bb], axis=2)
        beta_D = np.stack([beta_D_r, beta_D_g, beta_D_b], axis=2)
        degraded, direct, backscatter = self._degrade_image(img, dep, B, beta_D, wbalance)
        
        return degraded, direct, backscatter, wbalance
    
    def __call__(self, results):
        if 'dep_prefix' not in results:
            raise ValueError('the depth path must be specified.')
        
        # process depth
        dep_path = osp.join(results['dep_prefix'], results['ori_filename'])
        h, w = results['img_shape'][:2]
        depth = cv2.resize(cv2.imread(dep_path, cv2.IMREAD_GRAYSCALE), (w, h))
        results['depth'] = np.expand_dims(depth * 1.0 / 255., axis=0).astype(np.float32)
        depth = cv2.GaussianBlur(depth, (33, 33), 10) * 1.0 / 255.
        depth = self.multiply_depth * depth + self.add_depth
        
        ks = random.sample(self.coefs.keys(), self.num)
        
        results['syn_num'] = self.num  # synthesize num
        for i, k in enumerate(ks):
            coef = self.coefs[k]
            syn_img, t, b, w = self._synthesize(results['img'], depth, coef)
            results['syn_img%d' % i] = syn_img.astype(np.float32)
            results['direct%d' % i] = t.astype(np.float32)
            results['back%d' % i] = b.astype(np.float32)
            results['wbalance%d' % i] = w.astype(np.float32)
        
        return results