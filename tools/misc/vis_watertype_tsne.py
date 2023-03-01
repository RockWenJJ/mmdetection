from keras.preprocessing import image
from PIL import Image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from glob import glob
import os
from sklearn import manifold
import matplotlib.pyplot as plt
from mmdet.models.detectors import WaterFormerV4Encode, WaterFormerV3Encode
from mmcv.runner import load_checkpoint
from einops import rearrange
import torch

# model = VGG16(weights='imagenet', include_top=False)
checkpoint = '/home/wenjj/Documents/01_Projects/mmdetection/work_dirs/waterformerv4_uwcnn/epoch_1.pth'
gpu_id = 1
model = WaterFormerV4Encode()
checkpoint = load_checkpoint(model, checkpoint, map_location='cuda:%d'%gpu_id)
model = model.cuda('cuda:%d'%gpu_id)
avg = torch.nn.AdaptiveAvgPool2d(1).cuda('cuda:%d'%gpu_id)
colors = ['red', 'orange', 'blue', 'green', 'yellow', 'cyan']
labels = ["0", "1",  '2', '3', '4', '5']

# colors = ['red', 'green', 'cyan']
# labels = ['0',  '3', '4']

# colors = ['red', 'orange', 'blue', 'green']
# labels = ["0", "1",  '2', '3']

input_dir = '/home/wenjj/Documents/01_Projects/02_Papers/04_ICCV2023/water-type'

feats = []
for label in labels:
    sub_dir = os.path.join(input_dir, label)
    images = os.listdir(sub_dir)
    features = []
    for img in images[:30]:
        # img_path = "/home/wenjj/Documents/01_Projects/02_Papers/03_ICRA23/t-SNE/syrea/blue_00012_429.png"
        im = Image.open(os.path.join(input_dir, label, img))
        im = im.resize((256, 256))
        img_data = np.array(im)
        img_data = np.expand_dims(img_data, axis=0)
        # img_data = preprocess_input(img_data)
        img_data = img_data / 255.
        img_data = torch.from_numpy(img_data).float().to('cuda:%d'%gpu_id)
        img_data = rearrange(img_data, 'b h w c -> b c h w')
        feature = model.forward_img(img_data)
        # feature = model.latent[0](feature)
        embedding = model.latent[0].adaptor(torch.LongTensor([int(label)]).to('cuda:%d'%gpu_id))
        feature = embedding[..., None, None]
        # feature = embedding + np.rand_like(embedding.shape)
        # feature = avg(feature)
        # feature = feature[0, 127, ...]
        features.append(np.reshape(feature.cpu().data.numpy(), (1, -1)))
        # features = [np.reshape(feature.cpu().data.numpy(), (1, -1))]

    feat = np.concatenate(features, 0)
    # print(feat.shape)
    feats.append(feat)

tsne = manifold.TSNE(n_components=2, init='pca', random_state=500, angle=0.99)

X_tsnes = []
for i, feat in enumerate(feats):
    X_tsne = tsne.fit_transform(feat)
    X_tsnes.append(X_tsne)

from matplotlib import gridspec
plt.rc('font', family="Times New Roman")
fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(1, 1, height_ratios=[1])
plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
ax0 = plt.subplot(gs[0])

    # center = np.mean(X_tsne, 0)

for i, X_tsne in enumerate(X_tsnes):
    # if i == 3:
    #     X_tsne[X_tsne[:, 0] > 340] = None

    ax0.scatter(X_tsne[:, 0], X_tsne[:, 1], s=100, marker='o', color=colors[i], edgecolor='none', label=labels[i], alpha=1.0)
    # print(np.mean(X_tsne, 0))

plt.show()
# plt.xticks([])
# plt.yticks([])
# plt.xlim([-400, 400])
# plt.ylim([-400, 400])
# plt.axis('equal')
# plt.legend(loc=1)
# plt.savefig("tSNE_uw_imgs.pdf")
# plt.show()
# print("test")