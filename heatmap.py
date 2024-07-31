import numpy as np
from skimage import io
from glob import glob
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import random
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo
import torch.utils.data as data
import torch.optim as optim
import torch.nn.init
from torch.autograd import Variable
from IPython.display import clear_output

import time
import torch.autograd as autograd
import cv2 
interp = nn.Upsample(size=(256, 256), mode='bilinear')

_, _, _, pred_target2, _, atty, _, _ = net(image_patchess, image_patches)
pred2 = F.softmax(pred_target2, dim=1)
outs = interp(pred2)

# x_comp = 80
# y_comp = 20
# pred = outs[:, 1, x_comp, y_comp]

x_comp = 50
y_comp = 100
pred = outs[:, 4, x_comp, y_comp]
feature = outs
feature_grad = autograd.grad(pred, feature, allow_unused=True, retain_graph=True)[0]

grads = feature_grad  # 获取梯度
pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
# 此处batch size默认为1，所以去掉了第0维（batch size维）
pooled_grads = pooled_grads[0]
feature = feature[0]
# print("pooled_grads:", pooled_grads.shape)
# print("feature:", feature.shape)
# feature.shape[0]是指定层feature的通道数
for i in range(feature.shape[0]):
    feature[i, ...] *= pooled_grads[i, ...]

heatmap = feature.detach().cpu().numpy()
heatmap = np.mean(heatmap, axis=0)
heatmap1 = np.maximum(heatmap, 0)
heatmap1 /= np.max(heatmap1)
heatmap1 = cv2.resize(heatmap1, (256, 256))
# heatmap[heatmap < 0.7] = 0
heatmap1 = np.uint8(255 * heatmap1)
heatmap1 = cv2.applyColorMap(heatmap1, cv2.COLORMAP_JET)
heatmap1 = heatmap1[:, :, (2, 1, 0)]

fig = plt.figure()
fig.add_subplot(1, 2, 1)
image_patches = np.asarray(255 * torch.squeeze(image_patches).cpu(), dtype='uint8').transpose((1, 2, 0))
plt.imshow(image_patches)
plt.axis('off')
plt.gca().add_patch(plt.Rectangle((x_comp - 2, y_comp - 2), 2, 2, color='red', fill=False, linewidth=1))

fig.add_subplot(1, 2, 2)
plt.imshow(heatmap1)
plt.gca().add_patch(plt.Rectangle((x_comp - 2, y_comp - 2), 2, 2, color='red', fill=False, linewidth=1))
plt.axis('off')
plt.savefig('CSATAGAN_car.pdf', dpi=1200)
plt.show()