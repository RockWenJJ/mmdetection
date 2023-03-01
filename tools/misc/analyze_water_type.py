import numpy as np
import json as js
import os
import shutil
from tqdm import tqdm

input_dir = '/mnt/03_Data/nyudepthv2/uie/v2/synthesis_256_test'
input_file = '/home/wenjj/Documents/01_Projects/02_Papers/04_ICCV2023/water-type/results.txt'
images = os.listdir(input_dir)

out_dir = '/home/wenjj/Documents/01_Projects/02_Papers/04_ICCV2023/water-type'

file = open(input_file, 'r')

lines = file.readlines()
num = len(lines) // 2
for i in tqdm(range(num)):
    filename = lines[i*2].split('/')[-1][:-1]
    cls = lines[i*2+1].split('/')[-1][:-1]
    
    if not os.path.exists(os.path.join(out_dir, str(cls))):
        os.makedirs(os.path.join(out_dir, str(cls)), exist_ok=True)
    shutil.copy(os.path.join(input_dir, filename), os.path.join(out_dir, str(cls)))
