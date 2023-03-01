import cv2
import glob
import os
from tqdm import tqdm
import json
import numpy as np

# in_dir = "/mnt/03_Data/UWCNN/ALL/synthesis_256"
# out_dir = "/mnt/03_Data/UWCNN/ALL/synthesis_128"
#
# os.makedirs(out_dir, exist_ok=True)
#
# images = os.listdir(in_dir)
#
# h, w = 128, 128
#
#
# for i, image in tqdm(enumerate(images)):
#     basename = image[:-4]
#     im = cv2.imread(os.path.join(in_dir, image))
#     # h, w = im.shape[:2]
#     im = cv2.resize(im, (h, w))
#     # s = h // 2 - 128//2
#     # im = im[s:128, s:128, :]
#     # info = dict(filename=basename + '.png', height=h, width=w, id=i + 1)
#     cv2.imwrite(os.path.join(out_dir, basename + '.png'), im)


infos = json.load(open('/mnt/03_Data/nyudepthv2/uie/v2/test_infos.json'))

in_dir = "/mnt/03_Data/nyudepthv2/uie/v2/synthesis"
out_dir = "/mnt/03_Data/nyudepthv2/uie/v2/synthesis_256"

os.makedirs(out_dir, exist_ok=True)

h, w = 256, 256

for info in tqdm(infos):
    name = info['filename']
    img = cv2.imread(os.path.join(in_dir, name))
    img = cv2.resize(img, (h, w))
    cv2.imwrite(os.path.join(out_dir, name), img)