import os
import json as js
import cv2
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    # infos = js.load(open("/home/wenjj/Documents/01_Projects/mmdetection/data/synthesis/test_infos.json", "r"))
    print("test")
    input_dir = "/mnt/03_Data/nyudepthv2/uie/v1/synthesis_v1"
    names = os.listdir(input_dir)
    train_infos = []
    test_infos = []
    train_id, test_id = 0, 0
    for i, name in tqdm(enumerate(names)):
        # img = cv2.imread(os.path.join(input_dir, name))
        # h, w, _ = img.shape
        info = dict(filename=name,
                    height=460,
                    width=620,
                    id=i+1)
        rand = np.random.rand()
        if rand < 0.95:
            train_id += 1
            info['id'] = train_id
            train_infos.append(info)
        else:
            test_id += 1
            info['id'] = test_id
            test_infos.append(info)
    
    js.dump(train_infos, open("train_infos.json", "w"))
    js.dump(test_infos, open("test_infos.json", "w"))