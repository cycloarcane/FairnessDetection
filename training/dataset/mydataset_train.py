import os
import pandas as pd
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import albumentations as A
import cv2

from dataset.albu import IsotropicResize


def read(csv_folder):
    categories = set(f.split('_')[0] for f in os.listdir(csv_folder) if f.endswith('.csv'))
    data_dict = {}
    for category in categories:
        csv_real_file = os.path.join(csv_folder, f'{category}_real.csv')
        csv_fake_file = os.path.join(csv_folder, f'{category}_fake.csv')

        if os.path.exists(csv_real_file):
            real_images = pd.read_csv(csv_real_file)
            data_dict[f'{category}_real'] = real_images
        else:
            print(f'No real images for {category}')

        if os.path.exists(csv_fake_file):
            fake_images = pd.read_csv(csv_fake_file)
            data_dict[f'{category}_fake'] = fake_images
        else:
            print(f'No fake images for {category}')

    return data_dict


class thirdAugDataset(Dataset):
    def __init__(self, csv_folder):
        self.csv_folder = csv_folder
        self.categories = set(f.split('_')[0] for f in os.listdir(self.csv_folder) if f.endswith('.csv'))
        self.data_dict = read(self.csv_folder)
        self.index = 0
        self.transform = self.init_data_aug_method()
        self.len_list = pd.read_csv("/lab/kirito/data/CNNspot_test/datacsv/bicycle_fake.csv")
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
        return T.ToTensor()(img)

    def normalize(self, img):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)

    def data_aug(self, img):
        kwargs = {'image': img}
        transformed = self.transform(**kwargs)
        augmented_img = transformed['image']
        return augmented_img

    def __getitem__(self, idx):
        global real_label, fake_label
        if torch.is_tensor(idx):
            idx = idx.tolist()
        real_trans_list = []
        fake_trans_list = []

        real_list = ['airplane_real', 'train_real', 'pottedplant_real', 'motorbike_real', 'bird_real', 'sofa_real',
                     'cat_real', 'sheep_real', 'diningtable_real', 'cow_real', 'horse_real', 'bottle_real', 'car_real',
                     'bicycle_real', 'dog_real', 'person_real', 'tvmonitor_real', 'chair_real', 'boat_real', 'bus_real']
        fake_list = ['airplane_fake', 'train_fake', 'pottedplant_fake', 'motorbike_fake', 'bird_fake', 'sofa_fake',
                     'cat_fake', 'sheep_fake', 'diningtable_fake', 'cow_fake', 'horse_fake', 'bottle_fake', 'car_fake',
                     'bicycle_fake', 'dog_fake', 'person_fake', 'tvmonitor_fake', 'chair_fake', 'boat_fake', 'bus_fake']

        for self.index in range(20):
            real_image_list = self.data_dict[real_list[self.index]]
            fake_image_list = self.data_dict[fake_list[self.index]]
            
            real_img_path = real_image_list.iloc[idx, 0]
            fake_img_path = fake_image_list.iloc[idx, 0]

            real_img = Image.open(real_img_path).convert('RGB')
            fake_img = Image.open(fake_img_path).convert('RGB')

            real_img = np.array(real_img)
            fake_img = np.array(fake_img)

            real_image_trans = self.data_aug(real_img)
            fake_image_trans = self.data_aug(fake_img)

            real_trans = self.normalize(self.to_tensor(real_image_trans))
            fake_trans = self.normalize(self.to_tensor(fake_image_trans))
           
            real_trans_list.append(real_trans)
            fake_trans_list.append(fake_trans)
        real_label = np.array(real_image_list.iloc[idx, 1])
        fake_label = np.array(fake_image_list.iloc[idx, 1])
        # fake_label = np.tile(fake_label,(19,1))

        return {"real": (real_trans_list, real_label),
                "fake": (fake_trans_list, fake_label)}

    def __len__(self):
        return len(self.len_list)

    @staticmethod
    def collate_fn(batch):
        index = random.randint(0,20)

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
        fake_images, fake_labels = zip(
            *[data["fake"] for data in batch])
        # print(fake_labels)
        fake_labels = tuple(x.item() for x in fake_labels)

        real_images, real_labels = zip(
            *[data["real"] for data in batch])
        real_labels = tuple(x.item() for x in real_labels)

        # Stack the image, label, tensors for fake and real data
        fake_images_tensor = []
        real_images_tensor = []

        fake_images_tensor = torch.stack([torch.stack(item, dim=0) for item in fake_images], dim=0)
        real_images_tensor = torch.stack([torch.stack(item, dim=0) for item in real_images], dim=0)
        fake_labels = torch.LongTensor([item for sublist in fake_labels for item in [sublist] * 20])
        real_labels = torch.LongTensor([item for sublist in real_labels for item in [sublist] * 20])

        '''for idx, item in enumerate(fake_images):
            fake_images_tensor = torch.stack(item, dim=0)
        
        fake_labels = torch.LongTensor(fake_labels)
        fake_labels = torch.stack([fake_labels] * 20, dim=0)

        for idx,item in enumerate(real_images):
            real_images_tensor = torch.stack(item, dim=0)

        real_labels = torch.LongTensor(real_labels)
        real_labels = torch.stack([real_labels] * 20, dim=0)'''

        # Combine the fake and real tensors and create a dictionary of the tensors
        '''images = torch.cat([real_images, fake_images], dim=0)
            labels = torch.cat([real_labels, fake_labels], dim=0)'''

        '''fake_images_tensor_fin = fake_images_tensor[index]
        fake_labels_tensor_fin = fake_labels[index]
        print(fake_images_tensor_fin.shape)
        print(fake_labels_tensor_fin.shape)
        print(fake_labels_tensor_fin)

        select_tensor = [real_images_tensor[i] for i in range(20) if i != index]
        real_images_tensor_fin = torch.stack(select_tensor, dim=0)
        select_labels = [real_labels[v] for v in range(20) if v != index]
        real_labels_fin = torch.stack(select_labels, dim=0)
        print(real_images_tensor_fin.shape)
        print(real_labels_fin.shape)'''
        '''print(type(real_labels),real_labels.shape)
        print(type(fake_labels),fake_labels.shape)
        print(real_labels)
        print(fake_labels)'''


        data_dict = {
            'real_images': real_images_tensor,
            'real_labels': real_labels,
            'fake_images': fake_images_tensor,
            'fake_labels': fake_labels
        }

        return data_dict


'''# 使用示例
csv_folder = '/home/dell/Documents/Appledog/data/CNNspot/train/progan/datacsv'  # 替换为实际的CSV文件夹路径
dataset = thirdAugDataset(csv_folder)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=thirdAugDataset.collate_fn)

data_iter = iter(data_loader)
batch_data = next(data_iter)'''

