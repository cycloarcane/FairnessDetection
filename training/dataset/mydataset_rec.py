'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30

The code is designed for scenarios such as disentanglement-based methods where it is necessary to ensure an equal number of positive and negative samples.
'''

import torch
import random
import numpy as np
import csv
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import pickle
import os
import pandas as pd
from PIL import Image
import random
from torchvision import transforms as T

import albumentations as A

from dataset.albu import IsotropicResize


class thirdAugDataset(Dataset):
    def __init__(self, csv_real_file, csv_fake_file, csv_realrec_file, csv_fakerec_file):

        # Get real and fake image lists
        # Fix the label of real images to be 0 and fake images to be 1
        self.fake_image_list = pd.read_csv(csv_fake_file)
        self.real_image_list = pd.read_csv(csv_real_file)
        self.fake_rec_image_list = pd.read_csv(csv_fakerec_file)
        self.real_rec_image_list = pd.read_csv(csv_realrec_file)

        # self.num_to_sample = len(self.real_image_list)
        # self.newsample_fake_images = random.sample(self.fake_images_list, self.num_to_sample)

        self.transform = self.init_data_aug_method()

        # self.fake_imglist = [(img, label, 1) for img, label in zip(
        #     self.image_list, self.label_list) if label != 0]
        # self.real_imglist = [(img, label, 0) for img, label in zip(
        #     self.image_list, self.label_list) if label == 0]

    def init_data_aug_method(self):
        trans = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=[-10, 10], p=0.5),
            A.GaussianBlur(blur_limit=[3, 7], p=0.5),
            A.OneOf([
                IsotropicResize(max_side=224, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=224, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=224, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            ], p=1),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=[-0.1, 0.1], contrast_limit=[-0.1, 0.1]),
                A.FancyPCA(),
                A.HueSaturationValue()
            ], p=0.5),
            A.ImageCompression(quality_lower=40, quality_upper=100, p=0.5)
        ],
        )
        return trans

    def to_tensor(self, img):
        """
        Convert an image to a PyTorch tensor.
        """
        return T.ToTensor()(img)

    def normalize(self, img):
        """
        Normalize an image.
        """
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)

    def data_aug(self, img):
        """
        Apply data augmentation to an image, landmark, and mask.

        Args:
            img: An Image object containing the image to be augmented.
            landmark: A numpy array containing the 2D facial landmarks to be augmented.
            mask: A numpy array containing the binary mask to be augmented.

        Returns:
            The augmented image, landmark, and mask.
        """

        # Create a dictionary of arguments
        kwargs = {'image': img}

        transformed = self.transform(**kwargs)

        # Get the augmented image, landmark, and mask
        augmented_img = transformed['image']
        return augmented_img

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        fake_img_path = self.fake_image_list.iloc[idx, 0]
        fake_imgrec_path = self.fake_rec_image_list.iloc[idx, 0]
        '''if 'crop' in fake_img_path:
            fake_img_path = fake_img_path[3:]
        else:
            fake_img_path = fake_img_path[3:8] + 'crop_img/' + fake_img_path[8:]'''
        # real_idx = random.randint(0, len(self.real_image_list) - 1)
        real_img_path = self.real_image_list.iloc[idx, 0]
        real_imgrec_path = self.real_rec_image_list.iloc[idx, 0]
        '''if 'crop' in real_img_path:
            real_img_path = real_img_path[3:]
        else:
            real_img_path = real_img_path[3:8] + 'crop_img/' + real_img_path[8:]'''

        if fake_img_path != 'img_path':
            fake_img = Image.open(fake_img_path).convert('RGB')
            fake_imgrec = Image.open(fake_imgrec_path).convert('RGB')
            # fake_trans = self.transform(fake_img)
            fake_label = np.array(self.fake_image_list.iloc[idx, 1])
            fake_labelrec = np.array(self.fake_rec_image_list.iloc[idx, 1])
            # fake_spe_label = np.array(self.fake_image_list.iloc[idx, 7])

        if real_img_path != 'img_path':
            real_img = Image.open(real_img_path).convert('RGB')
            real_imgrec = Image.open(real_imgrec_path).convert('RGB')
            # real_trans = self.transform(real_img)
            real_label = np.array(self.real_image_list.iloc[idx, 1])
            real_labelrec = np.array(self.real_rec_image_list.iloc[idx, 1])
            # real_spe_label = np.array(self.real_image_list.iloc[real_idx, 1])
        fake_img = np.array(fake_img)
        real_img = np.array(real_img)
        fake_imgrec = np.array(fake_imgrec)
        real_imgrec = np.array(real_imgrec)
        
        if fake_img.shape[:2] != (224, 224):
            fake_img = cv2.resize(fake_img, (224, 224), interpolation=cv2.INTER_AREA)
        
        if real_img.shape[:2] != (224, 224):
            real_img = cv2.resize(real_img, (224, 224), interpolation=cv2.INTER_AREA)
            
        if fake_imgrec.shape[:2] != (224, 224):
            fake_imgrec = cv2.resize(fake_imgrec, (224, 224), interpolation=cv2.INTER_AREA)
        
        if real_imgrec.shape[:2] != (224, 224):
            real_imgrec = cv2.resize(real_imgrec, (224, 224), interpolation=cv2.INTER_AREA)
        
        '''fake_image_trans = self.data_aug(fake_img)
        real_image_trans = self.data_aug(real_img)
        fake_imgrec_trans = self.data_aug(fake_imgrec)
        real_imgrec_trans = self.data_aug(real_imgrec)'''

        # To tensor and normalize for fake and real images
        fake_trans = self.normalize(self.to_tensor(fake_img))
        fakerec_trans = self.normalize(self.to_tensor(fake_imgrec))
        real_trans = self.normalize(self.to_tensor(real_img))
        realrec_trans = self.normalize(self.to_tensor(real_imgrec))

        return {"fake": (fake_trans, fake_label),
                "real": (real_trans, real_label),
                "fakerec": (fakerec_trans, fake_labelrec),
                "realrec": (realrec_trans, real_labelrec)}

        '''return {"fake": (fake_trans, fake_label, fake_spe_label),
                "real": (real_trans, real_label, real_spe_label)}'''

    def __len__(self):
        return len(self.fake_image_list)

    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor, the label tensor,
                        the landmark tensor, and the mask tensor.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Separate the image, label,  tensors for fake and real data
        '''fake_images, fake_labels, fake_spe_labels = zip(
            *[data["fake"] for data in batch])'''
        fake_images, fake_labels = zip(
            *[data["fake"] for data in batch])
        fake_rec_images, fake_rec_labels = zip(
            *[data["fakerec"] for data in batch]
        )
        # print(fake_labels)
        fake_labels = tuple(x.item() for x in fake_labels)
        fake_rec_labels = tuple(x.item() for x in fake_rec_labels)
        # fake_spe_labels = tuple(x.item() for x in fake_spe_labels)

        '''real_images, real_labels, real_spe_labels = zip(
            *[data["real"] for data in batch])'''
        real_images, real_labels = zip(
            *[data["real"] for data in batch])
        real_rec_images, real_rec_labels = zip(
            *[data["realrec"] for data in batch]
        )
        real_labels = tuple(x.item() for x in real_labels)
        real_rec_labels = tuple(x.item() for x in real_rec_labels)
        # real_spe_labels = tuple(x.item() for x in real_spe_labels)

        # Stack the image, label, tensors for fake and real data
        fake_images = torch.stack(fake_images, dim=0)
        fake_labels = torch.LongTensor(fake_labels)
        fake_rec_images = torch.stack(fake_rec_images, dim=0)
        fake_rec_labels = torch.LongTensor(fake_rec_labels)
        # fake_spe_labels = torch.LongTensor(fake_spe_labels)
        # fake_fair_labels = torch.LongTensor(fake_fair_labels)

        real_images = torch.stack(real_images, dim=0)
        real_labels = torch.LongTensor(real_labels)
        real_rec_images = torch.stack(real_rec_images, dim=0)
        real_rec_labels = torch.LongTensor(real_rec_labels)
        # real_spe_labels = torch.LongTensor(real_spe_labels)
        # real_fair_labels = torch.LongTensor(real_fair_labels)

        # Combine the fake and real tensors and create a dictionary of the tensors
        images = torch.cat([real_images, fake_images], dim=0)
        labels = torch.cat([real_labels, fake_labels], dim=0)
        # spe_labels = torch.cat([real_spe_labels, fake_spe_labels], dim=0)
        # fair_labels = torch.cat([real_fair_labels, fake_fair_labels], dim=0)

        data_dict = {
            'image': images,
            'label': labels,
            # 'label_spe': spe_labels,
            'real_images': real_images,
            'real_labels': real_labels,
            'fake_images': fake_images,
            'fake_labels': fake_labels,
            'real_rec_images': real_rec_images,
            'real_rec_labels': real_rec_labels,
            'fake_rec_images': fake_rec_images,
            'fake_rec_labels': fake_rec_labels
        }
        return data_dict


'''csv_folder1 = '/home/dell/Documents/Appledog/data/CNNspot/train/realtrain.csv'  # 替换为实际的CSV文件夹路径
csv_folder2 = '/home/dell/Documents/Appledog/data/CNNspot/train/faketrain.csv'
dataset = thirdAugDataset(csv_folder1, csv_folder2)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=thirdAugDataset.collate_fn)

data_iter = iter(data_loader)
batch_data = next(data_iter)'''
