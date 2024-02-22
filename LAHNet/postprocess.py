# -*- coding: utf-8 -*-
import numpy as np
import cv2
import torchvision.transforms as transforms
import os

from utils import check_mkdir

# assert dataset_name in ['MICHE', 'CASIA-Iris-M1', 'CASIA-iris-distance', 'UBIRIS.v2']
# give 'MICHE' for example
dataset_name = 'MICHE'
model_name = 'lahnet'
time_dir = 'time_logging'
save_dir = os.path.join('./experiments/', dataset_name, model_name, time_dir, 'checkpoints')
print(save_dir)

test_dir = os.path.join('./iris-datasets/' + dataset_name + '/test/image')
mask_dir = os.path.join(save_dir, 'SegmentationClass')
out_dir_outer = os.path.join(save_dir, 'Outer_Boundary')
out_dir_inner = os.path.join(save_dir, 'Inner_Boundary')

result_vis_save_dir = os.path.join(save_dir, 'Result_Vis')
check_mkdir(result_vis_save_dir)
result_vis_circle_save_dir = os.path.join(save_dir, 'Result_Vis_Circle')
check_mkdir(result_vis_circle_save_dir)

img_list = os.listdir(out_dir_outer)

if dataset_name == 'CASIA-iris-distance':
    pix = '.JPEG'

if dataset_name == 'MICHE':
    pix = '.JPEG'
if dataset_name == 'CASIA-Iris-M1':
    pix = '.JPG'
if dataset_name == 'UBIRIS.v2':
    pix = '.JPEG'

for idx, i in enumerate(img_list):
    print(i)
    img_name = i[:-4] + pix
    
    image = cv2.imread(os.path.join(test_dir, img_name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(os.path.join(mask_dir, i), cv2.IMREAD_GRAYSCALE)
    outer = cv2.imread(os.path.join(out_dir_outer, i), cv2.IMREAD_GRAYSCALE)
    inner = cv2.imread(os.path.join(out_dir_inner, i), cv2.IMREAD_GRAYSCALE)

    h, w = mask.shape[0], mask.shape[1]
    mask_3d_color = np.zeros((h, w, 3), dtype='uint8')
    mask_3d_color[mask[:, :] == 255] = [100, 0, 0]
    img = image.astype('uint8')
    result_vis = cv2.add(img, mask_3d_color)
    result_vis = transforms.ToPILImage()(result_vis)

    result_vis.save(os.path.join(result_vis_save_dir, img_name + '.png'))

    mask_iris_3d_color = np.zeros((h, w, 3), dtype='uint8')
    mask_iris_3d_color[outer[:, :] == 255] = [0, 0, 255]
    result_vis_circle = cv2.add(img, mask_iris_3d_color)

    mask_pupil_3d_color = np.zeros((h, w, 3), dtype='uint8')
    mask_pupil_3d_color[inner[:, :] == 255] = [0, 255, 0]
    result_vis_circle = cv2.add(result_vis_circle, mask_pupil_3d_color)
    result_vis_circle = transforms.ToPILImage()(result_vis_circle)
    result_vis_circle.save(os.path.join(result_vis_circle_save_dir, i + '.png'))