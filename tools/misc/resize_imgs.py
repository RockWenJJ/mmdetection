import cv2
import glob
import os
from tqdm import tqdm
import json
import numpy as np

in_dir = "/mnt/03_Data/UWCNN/ALL/image"
out_dir = "/mnt/03_Data/UWCNN/ALL/image_128"

os.makedirs(out_dir, exist_ok=True)

images = os.listdir(in_dir)

h, w = 224, 224


for i, image in tqdm(enumerate(images)):
    basename = image[:-4]
    im = cv2.imread(os.path.join(in_dir, image))
    # h, w = im.shape[:2]
    im = cv2.resize(im, (h, w))
    s = h // 2 - 128//2
    im = im[s:128, s:128, :]
    # info = dict(filename=basename + '.png', height=h, width=w, id=i + 1)
    cv2.imwrite(os.path.join(out_dir, basename + '.png'), im)