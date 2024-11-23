import csv
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pickle
import os
import pandas as pd
from PIL import Image
import random
import albumentations as A
import cv2
from torchvision import transforms as T

class Pixelate(A.ImageOnlyTransform):
    def __init__(self, scale_factor, always_apply=False, p=0.5):
        super(Pixelate, self).__init__(always_apply=always_apply, p=p)
        self.scale_factor = scale_factor

    def apply(self, img, **params):
        height, width, channels = img.shape
        # 将图像缩小
        small = cv2.resize(img, (int(width / self.scale_factor), int(height / self.scale_factor)), interpolation=cv2.INTER_NEAREST)
        # 然后重新放大
        pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
        return pixelated
class JpegCompression(A.ImageOnlyTransform):
    def __init__(self, quality, p=1.0):
        super(JpegCompression, self).__init__(p=p)
        self.quality = quality  # JPEG 压缩质量

    def apply(self, img, **params):
        # 将图像编码为 JPEG，然后解码，以模拟压缩
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        _, encoded_img = cv2.imencode('.jpg', img, encode_param)  # JPEG 编码
        compressed_img = cv2.imdecode(encoded_img, 1)  # JPEG 解码
        return compressed_img
    
class RandomGridDropout(A.ImageOnlyTransform):
    def __init__(self, num_points_range=(10, 20), point_size_range=(5, 10), always_apply=False, p=0.5):
        super(RandomGridDropout, self).__init__(always_apply, p)
        self.num_points_range = num_points_range
        self.point_size_range = point_size_range

    def apply(self, image, **params):
        height, width = image.shape[:2]
        num_points = np.random.randint(*self.num_points_range)
        point_size = np.random.randint(*self.point_size_range)

        for _ in range(num_points):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)

            x_start = max(0, x - point_size)
            x_end = min(width, x + point_size)
            y_start = max(0, y - point_size)
            y_end = min(height, y + point_size)

            image[y_start:y_end, x_start:x_end] = 0

        #     image = F.cutout(image, y_start, x_start, y_end - y_start, x_end - x_start, fill_value=0)

        return image
class ImageDataset_Test_Robust(Dataset):
    # def __init__(self, csv_file, img_size, filter_size, test_set):
    def __init__(self, csv_file, robustness, level):

        self.transform = transforms.Compose([
            transforms.Resize((256, 256))])
        self.img_list = pd.read_csv(csv_file)
        if robustness == 'gausionNoise':
            varlimit = [(5, 10), (15,20), (25, 30), (35,40), (45,50)]
            self.robus = A.GaussNoise(var_limit=varlimit[level-1], mean=0, p=1.0)
        elif robustness == 'gausionBlur':
            blurlimit = [(3,3), (5,5), (7, 7), (9,9), (11,11)]
            self.robus = A.GaussianBlur(blur_limit=blurlimit[level-1], p=1.0)
        elif robustness == 'block-wiseNoise':
            ratiolist = [0.1, 0.2, 0.3, 0.4, 0.5]
            minlist = [10, 15, 20, 25, 30]
            maxlist = [10, 15, 20, 25, 30]
            self.robus = A.GridDropout(ratio = ratiolist[level-1], unit_size_min=minlist[level-1], unit_size_max=maxlist[level-1], p = 1.0)
        elif robustness == 'newblock-wiseNoise':
            numlist = [(1,5), (6,10), (11,15), (16,20), (21,25)]
            sizelist = [(1,3), (2,4), (3,6), (4,7), (5,8)]
            size=(1,1)
            self.robus = RandomGridDropout(num_points_range=numlist[level-1], point_size_range=sizelist[level-1],p=1.0)
        elif robustness == 'pixLation':
            scalelist = [1, 2, 3, 4, 5]
            self.robus = Pixelate(scale_factor=scalelist[level-1], p = 1.0)
        elif robustness == 'colorContrast':
            conlist = [0.1, 0.2, 0.3, 0.4, 0.5]
            self.robus = A.RandomBrightnessContrast(contrast_limit=conlist[level-1], p = 1.0)
        elif robustness == 'colorSaturation':
            satlist = [(-10,10),(-20,20), (-30,30), (-40,40),(-50,50)]
            huevallist= [(-5,5),(-10,10),(-15,15),(-20,20),(-25,25)]
            self.robus = A.HueSaturationValue(hue_shift_limit=huevallist[level-1],sat_shift_limit=satlist[level-1],val_shift_limit=huevallist[level-1], p=1.0)
        elif robustness == 'imgCompression':
            qualist = [90, 75, 60, 45, 30]
            self.robus = JpegCompression(quality=qualist[level-1], p=1.0)
        elif robustness == 'affineTransformation':
            limitlist = [0.1, 0.2, 0.3, 0.4, 0.5]
            rotlist = [10, 20, 30, 40, 50]
            self.robus = A.ShiftScaleRotate(shift_limit=limitlist[level-1], scale_limit=limitlist[level-1], rotate_limit=rotlist[level-1], p=1.0)

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
        
    
        robused = self.robus(**kwargs)
        
        # Get the augmented image, landmark, and mask
        augmented_img = robused['image']
       

        return augmented_img
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        path = self.img_list.iloc[index,0]
        if 'crop' in path:
            path = path[3:]
        else:
            path = path[3:8] + 'crop_img/' + path[8:]
        img = Image.open(path)
        label = np.array(self.img_list.iloc[index,1])
        img = self.transform(img)
        img = np.array(img)
        img = self.data_aug(img)
        img = self.normalize(self.to_tensor(img))
        data_dict = {}
        data_dict['image'] = img
        data_dict['label'] = label

        return data_dict

    def __len__(self):
        return len(self.img_list)
