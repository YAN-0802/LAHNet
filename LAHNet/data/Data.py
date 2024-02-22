import os
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


root = 'iris-datasets/'

def make_dataset_list(dataset_name, mode):

    assert dataset_name in ['MICHE_UBIRIS', 'MICHE', 'CASIA-Iris-M1', 'CASIA-iris-distance', 'UBIRIS.v2', ]
    assert mode in ['train', 'val', 'test']

    if mode == 'train':
        train_data_path = root + dataset_name + '/train'
        data_path = {
            'images_path': os.path.join(train_data_path, 'image'),
            'masks_path': os.path.join(train_data_path, 'SegmentationClass'),
            'irises_edge_path': os.path.join(train_data_path, 'iris_edge'),
            'irises_edge_mask_path': os.path.join(train_data_path, 'iris_edge_mask'),
            'pupils_edge_path': os.path.join(train_data_path, 'pupil_edge'),
            'pupils_edge_mask_path': os.path.join(train_data_path, 'pupil_edge_mask'),
            'heatmaps_path': os.path.join(train_data_path, 'loc_result'),
            # 'gauss_heatmaps_path': os.path.join(train_data_path, 'gauss_heatmap'),
            # 'dis_heatmaps_path': os.path.join(train_data_path, 'dis_heatmap'),
        }

        images_filenames_list = list(os.listdir(data_path['images_path']))
        random.seed(42)

        return data_path, images_filenames_list

    if mode == 'test':
        test_data_path = root + dataset_name + '/test'
        data_path = {
            'images_path': os.path.join(test_data_path, 'image'),
            'masks_path': os.path.join(test_data_path, 'SegmentationClass'),
            'irises_edge_path': os.path.join(test_data_path, 'iris_edge'),
            'irises_edge_mask_path': os.path.join(test_data_path, 'iris_edge_mask'),
            'pupils_edge_path': os.path.join(test_data_path, 'pupil_edge'),
            'pupils_edge_mask_path': os.path.join(test_data_path, 'pupil_edge_mask'),
            'heatmaps_path': os.path.join(test_data_path, 'loc_result')
        }
        test_filenames_list = list(os.listdir(data_path['images_path']))
        return data_path, test_filenames_list

    else:
        val_data_path = root + dataset_name + '/test'
        data_path = {
            'images_path': os.path.join(val_data_path, 'image'),
            'masks_path': os.path.join(val_data_path, 'SegmentationClass'),
            'irises_edge_path': os.path.join(val_data_path, 'iris_edge'),
            'irises_edge_mask_path': os.path.join(val_data_path, 'iris_edge_mask'),
            'pupils_edge_path': os.path.join(val_data_path, 'pupil_edge'),
            'pupils_edge_mask_path': os.path.join(val_data_path, 'pupil_edge_mask'),
            'heatmaps_path': os.path.join(val_data_path, 'loc_result')
        }
        val_filenames_list = list(os.listdir(data_path['images_path']))
        return data_path, val_filenames_list


class mulDataset(Dataset):
    '''
    args:
        dataset_name(str): support for 'MICHE', 'CASIA-Iris-M1', 'CASIA-iris-distance', 'UBIRIS.v2'
        mode(str): 'train', 'val', 'test'
        transform(dict): {'train': train_augment, 'test': test_augment}

    return(dict): {
        'image': aug_img,
        'mask': aug_mask,
        'iris_edge': aug_iris_edge
        'iris_edge_mask': aug_iris_edge_mask
        'pupil_edge': aug_pupil_edge
        'pupil_edge_mask': aug_pupil_edge_mask
    }
    '''
    def __init__(self, dataset_name, mode, transform=None):
        self.dataset_name = dataset_name
        self.mode = mode
        self.transform = transform
        self.data_path, self.data_list = make_dataset_list(dataset_name, mode)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        if self.mode == 'test':
            image_name = self.data_list[idx].split('.')[0]
            image = Image.open(os.path.join(self.data_path['images_path'], self.data_list[idx]))
            mask = Image.open(os.path.join(self.data_path['masks_path'], image_name + '.png'))
            iris_edge = Image.open(os.path.join(self.data_path['irises_edge_path'], image_name + '.png'))
            iris_edge_mask = Image.open(os.path.join(self.data_path['irises_edge_mask_path'], image_name + '.png'))
            pupil_edge = Image.open(os.path.join(self.data_path['pupils_edge_path'], image_name + '.png'))
            pupil_edge_mask = Image.open(os.path.join(self.data_path['pupils_edge_mask_path'], image_name + '.png'))
            heatmap = Image.open(os.path.join(self.data_path['heatmaps_path'], image_name + '.png'))
            # if image.size != heatmap.size:
            #    heatmap = heatmap.resize(image.size, resample=Image.BILINEAR)

            if self.transform is not None:
                image = np.asarray(image)
                heatmap = np.asarray(heatmap)
                # aug_data = self.transform(image=image)
                aug_data = self.transform(image=image, mask=heatmap)
                aug_image = aug_data['image']
                aug_heatmap = aug_data['mask']
                image = Image.fromarray(aug_image)
                heatmap = Image.fromarray(aug_heatmap)

            image = transforms.ToTensor()(image)
            mask = transforms.ToTensor()(mask)
            iris_edge = transforms.ToTensor()(iris_edge)
            iris_edge_mask = transforms.ToTensor()(iris_edge_mask)
            pupil_edge = transforms.ToTensor()(pupil_edge)
            pupil_edge_mask = transforms.ToTensor()(pupil_edge_mask)
            heatmap = transforms.ToTensor()(heatmap)
            return {
                'image_name': image_name,
                'image': image,
                'mask': mask,
                'iris_edge': iris_edge,
                'iris_edge_mask': iris_edge_mask,
                'pupil_edge': pupil_edge,
                'pupil_edge_mask': pupil_edge_mask,
                'heatmap': heatmap
            }
        
        image_name = self.data_list[idx].split('.')[0]
        image = Image.open(os.path.join(self.data_path['images_path'], self.data_list[idx]))
        mask = Image.open(os.path.join(self.data_path['masks_path'], image_name + '.png'))
        iris_edge = Image.open(os.path.join(self.data_path['irises_edge_path'], image_name + '.png'))
        iris_edge_mask = Image.open(os.path.join(self.data_path['irises_edge_mask_path'], image_name + '.png'))
        pupil_edge = Image.open(os.path.join(self.data_path['pupils_edge_path'], image_name + '.png'))
        pupil_edge_mask = Image.open(os.path.join(self.data_path['pupils_edge_mask_path'], image_name + '.png'))
        heatmap = Image.open(os.path.join(self.data_path['heatmaps_path'], image_name + '.png'))
        # if image.size != heatmap.size:
        #     heatmap = heatmap.resize(image.size, resample=Image.BILINEAR)
        # gauss_loc = Image.open(os.path.join(self.data_path['gauss_heatmaps_path'], image_name + '.png'))


        if self.transform is not None:
            image = np.asarray(image)
            mask = np.asarray(mask)
            if np.max(mask) < 2:
                mask = mask * 255
            iris_edge = np.asarray(iris_edge)
            iris_edge_mask = np.asarray(iris_edge_mask)
            pupil_edge = np.asarray(pupil_edge)
            pupil_edge_mask = np.asarray(pupil_edge_mask)
            heatmap = np.asarray(heatmap)
            # loc = get_loc(iris_edge, pupil_edge)
            # dis_loc = np.asarray(dis_loc)

            mask_list = [mask, iris_edge, iris_edge_mask, pupil_edge, pupil_edge_mask, heatmap]
            # mask_list = [mask, iris_edge, iris_edge_mask, pupil_edge, pupil_edge_mask]

            aug_data = self.transform(image=image, masks=mask_list)
            aug_image, aug_mask_list = aug_data['image'], aug_data['masks']
            
            image = Image.fromarray(aug_image)
            mask = Image.fromarray(aug_mask_list[0])
            iris_edge = Image.fromarray(aug_mask_list[1])
            iris_edge_mask = Image.fromarray(aug_mask_list[2])
            pupil_edge = Image.fromarray(aug_mask_list[3])
            pupil_edge_mask = Image.fromarray(aug_mask_list[4])
            heatmap = Image.fromarray(aug_mask_list[5])
            # dis_loc = Image.fromarray(aug_mask_list[6])

        aug_image = transforms.ToTensor()(image)
        aug_mask = transforms.ToTensor()(mask)
        aug_iris_edge = transforms.ToTensor()(iris_edge)
        aug_iris_edge_mask = transforms.ToTensor()(iris_edge_mask)
        aug_pupil_edge = transforms.ToTensor()(pupil_edge)
        aug_pupil_edge_mask = transforms.ToTensor()(pupil_edge_mask)
        aug_heatmap = transforms.ToTensor()(heatmap)
        # aug_dis_loc = TF.ToTensor()(dis_loc)


        return {
            'image_name': image_name,
            'image': aug_image,
            'mask': aug_mask,
            'iris_edge': aug_iris_edge,
            'iris_edge_mask': aug_iris_edge_mask,
            'pupil_edge': aug_pupil_edge,
            'pupil_edge_mask': aug_pupil_edge_mask,
            'heatmap': aug_heatmap
            # 'dis_loc': aug_dis_loc,
        }