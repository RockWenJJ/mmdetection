from mmdet.models import SwinTransformer
from PIL import Image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from glob import glob
import os
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from mmcv.utils import ConfigDict
from mmdet.models import ConvTransformerEncoder

pretrained = "/home/wenjj/Documents/01_Projects/mmdetection/work_dirs/unet2_swin-syn_uie-percep-back/epoch_150.pth"
init_cfg = ConfigDict(type='Pretrained', checkpoint=pretrained)
# swin = SwinTransformer(pretrained=pretrained)
swin = SwinTransformer(convert_weights=True, init_cfg=init_cfg)
# swin.self.load_state_dict(state_dict, False)
swin.init_weights()
swin.eval()

img_dir = "/home/wenjj/Documents/01_Projects/02_Papers/03_ICRA23/t-SNE/real/random"
imgs = glob(os.path.join(img_dir, "*.png"))
out_dir = "/home/wenjj/Documents/01_Projects/02_Papers/03_ICRA23/t-SNE/real/swin_feats"

for img in tqdm(imgs):
    basename = os.path.basename(img)
    im = Image.open(img)
    im = im.resize((224, 224))
    # img_data = np.array(im)
    img_tensor = transforms.ToTensor()(im)
    img_tensor = img_tensor[None, ...]
    swin_feats = swin(img_tensor)
    for i, swin_feat in enumerate(swin_feats):
        out_d = os.path.join(out_dir, f'{i}')
        os.makedirs(out_d, exist_ok=True)
        feat = torch.mean(torch.sigmoid(swin_feat), 1).detach().cpu().squeeze().data
        plt.imshow(feat)
        plt.savefig(os.path.join(out_d, basename))
        plt.close()
    # print(vgg_feats)

