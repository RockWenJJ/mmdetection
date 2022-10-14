import cv2
import json
import os
from tqdm import tqdm

input_ann_file = '/mnt/03_Data/DUO/DUO_glnet/annotations/instances_test.json'
out_ann_file = '/mnt/03_Data/DUO/DUO_glnet/annotations/instances_test_crop256x256.json'
input_img_dir = '/mnt/03_Data/DUO/DUO_glnet/images/test'
out_img_dir = '/mnt/03_Data/DUO/DUO_glnet/images/test_crop256x256'
ori_shape = (288, 512) # h, w
crop_size = (256, 256)
margin_h, margin_w = 288 - 256, 512 - 256
offset_h = margin_h // 2
offset_w = margin_w // 2
crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]


ann_file = json.load(open(input_ann_file, 'r'))
img_infos, ann_infos = [], []

for img_info in ann_file['images']:
    img_info['height'] = crop_size[0]
    img_info['width'] = crop_size[1]
    img_infos.append(img_info)

ann_id = 1
for ann_info in ann_file['annotations']:
    new_ann_info = dict(area=ann_info['area'],
                        image_id=ann_info['image_id'],
                        category_id=ann_info['category_id'],
                        iscrowd=ann_info['iscrowd'],
                        ignore=ann_info['ignore'])
    
    # parse segmentation
    segmentation = []
    for i, seg in enumerate(ann_info['segmentation']):
        pt = seg - crop_x1 if i//2==0 else seg - crop_y1
        segmentation.append(pt)
        ignore = False
        ignore = True if (pt < 0 or pt >= crop_size[1]) and i//2==0 else ignore
        ignore = True if (pt < 0 or pt >= crop_size[0]) and i//2!=0 else ignore
    
    # parse bbox
    ori_bbox = ann_info['bbox']
    bbox = [ori_bbox[0] - crop_x1, ori_bbox[1] - crop_y1] + ori_bbox[2:]
    
    if not ignore:
        new_ann_info['segmentation'] = segmentation
        new_ann_info['bbox'] = bbox
        new_ann_info['id'] = ann_id
        ann_id += 1
        ann_infos.append(new_ann_info)

out = dict(images=img_infos,
           annotations=ann_infos,
           categories=ann_file['categories'])

json.dump(out, open(out_ann_file, 'w'))


## crop images
imgs = os.listdir(input_img_dir)
os.makedirs(out_img_dir, exist_ok=True)
for img in tqdm(imgs):
    im = cv2.imread(os.path.join(input_img_dir, img))
    out_img = im[crop_y1:crop_y2, crop_x1:crop_x2, ...]
    cv2.imwrite(os.path.join(out_img_dir, img), out_img)