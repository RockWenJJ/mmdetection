import numpy as np
import json
from glob import glob

# input_jsons = ['/mnt/03_Data/UIEB/UIEB_data/train_infos.json',
#                '/mnt/03_Data/EUVP/Paired/underwater_dark/train_infos.json',
#                '/mnt/03_Data/EUVP/Paired/underwater_imagenet/train_infos.json',
#                '/mnt/03_Data/EUVP/Paired/underwater_scenes/train_infos.json',
#                '/mnt/03_Data/nyudepthv2/uie/v2/train_infos.json']
#
# output_json = '/home/wenjj/Documents/01_Projects/mmdetection/data/syn_real/train_infos.json'

input_jsons = ['/mnt/03_Data/UIEB/UIEB_data/test_infos.json',
               '/mnt/03_Data/EUVP/Paired/underwater_dark/test_infos.json',
               '/mnt/03_Data/EUVP/Paired/underwater_imagenet/test_infos.json',
               '/mnt/03_Data/EUVP/Paired/underwater_scenes/test_infos.json',
               '/mnt/03_Data/nyudepthv2/uie/v2/test_infos.json']

output_json = '/home/wenjj/Documents/01_Projects/mmdetection/data/syn_real/test_infos.json'


combined_infos = []

id = 0
for input_json in input_jsons:
    infos = json.load(open(input_json, 'r'))
    for info in infos:
        id += 1
        info['id'] = id
    combined_infos.extend(infos)
    
json.dump(combined_infos, open(output_json, 'w'))