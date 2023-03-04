import cv2
import glob
import os
from tqdm import tqdm
import json
import numpy as np

in_dir = "/mnt/03_Data/UIE_Benchmark/Raw/all_ref"
out_dir = "/mnt/03_Data/UIE_Benchmark/Raw/all_ref_div32"
json_out_dir = "/home/wenjj/Documents/01_Projects/02_Papers/04_ICCV2023/videos/video4"
out_json = False


os.makedirs(out_dir, exist_ok=True)
train_infos = []
test_infos = []

images = os.listdir(in_dir)

for i, image in tqdm(enumerate(images)):
    if image.endswith('.png') or image.endswith('.jpg'):
        basename = image[:-4]
        im = cv2.imread(os.path.join(in_dir, image))
        h, w = im.shape[:2]
        h = (h // 32) * 32
        w = (w // 32) * 32
        info = dict(filename=basename+'.png', height=h, width=w, id=i+1)
        cv2.imwrite(os.path.join(out_dir, basename+'.png'), im)
        rand = np.random.rand()
        if rand < 1.:
            test_infos.append(info)
        else:
            train_infos.append(info)
        
if out_json:
    json.dump(train_infos, open(os.path.join(json_out_dir, 'train_infos.json'), 'w'))
    json.dump(test_infos, open(os.path.join(json_out_dir, 'test_infos.json'), 'w'))
    print('Output json file successfully!')