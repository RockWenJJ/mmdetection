import cv2
import glob
import os
from tqdm import tqdm
import json
import numpy as np

in_dir = "/home/wenjj/Documents/01_Projects/mmdetection/data/dpdd/image"
out_dir = "/home/wenjj/Documents/01_Projects/mmdetection/data/dpdd/image"
json_out_dir = "/home/wenjj/Documents/01_Projects/mmdetection/data/dpdd"
out_json = True


os.makedirs(out_dir, exist_ok=True)
train_infos = []
test_infos = []

images = os.listdir(in_dir)

for i, image in tqdm(enumerate(images)):
    basename = image.split('.')[0]
    im = cv2.imread(os.path.join(in_dir, image))
    h, w = im.shape[:2]
    info = dict(filename=basename+'.png', height=h, width=w, id=i+1)
    # cv2.imwrite(os.path.join(out_dir, basename+'.png'), im)
    rand = np.random.rand()
    if rand < 0.02:
        test_infos.append(info)
    else:
        train_infos.append(info)
        
if out_json:
    json.dump(train_infos, open(os.path.join(json_out_dir, 'train_infos.json'), 'w'))
    json.dump(test_infos, open(os.path.join(json_out_dir, 'test_infos.json'), 'w'))
    print('Output json file successfully!')