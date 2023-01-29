from mmcv.cnn import VGG
from PIL import Image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from glob import glob
import os
from torchvision import transforms

vgg = VGG(depth=16, with_last_pool=False)
pretrained = "https://download.pytorch.org/models/vgg16-397923af.pth"
# load from pretrained ckpt
vgg.init_weights(pretrained)
vgg.eval()

img_dir = "/home/wenjj/Documents/01_Projects/02_Papers/03_ICRA23/t-SNE/real/random"
imgs = glob(os.path.join(img_dir, "*.png"))

for img in imgs:
    im = Image.open(img)
    im = im.resize((224, 224))
    img_data = np.array(im)
    img_tensor = transforms.ToTensor(img_data)
    
    
