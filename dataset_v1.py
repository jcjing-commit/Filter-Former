import random

from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
import torch.multiprocessing
import json
import copy
# import imgaug.augmenters as iaa
# from perlin import rand_perlin_2d_np

torch.multiprocessing.set_sharing_strategy('file_system')


import numpy as np
import cv2
from torchvision import transforms

def simulate_lighting(image, light_direction):
    """
    模拟不同方向的光源
    :param image: 输入图像 (H, W, C)
    :param light_direction: 光源方向 (0-360度)
    :return: 添加光照效果的图像
    """
    # 将图像转换为浮点数
    image = image.astype(np.float32) / 255.0
    
    # 创建一个光照遮罩
    h, w, _ = image.shape
    mask = np.ones((h, w), dtype=np.float32)
    center_x, center_y = w // 2, h // 2

    # 随机添加偏置
    bias_x = np.random.randint(-150, 151)  # 偏置范围为 -50 到 50
    bias_y = np.random.randint(-150, 151)  # 偏置范围为 -50 到 50

    # 更新中心点
    center_x += bias_x
    center_y += bias_y

    # 确保中心点在图像范围内
    center_x = max(0, min(center_x, w - 1))
    center_y = max(0, min(center_y, h - 1))


    # 根据光源方向生成遮罩
    light_rate = np.random.uniform(0.7, 1.4)
    for i in range(h):
        for j in range(w):
            angle = np.arctan2(center_y - i, center_x - j) * 180 / np.pi
            if (angle + 90) % 360 <= light_direction % 360:
                mask[i, j] = light_rate
    
    # 应用光照遮罩
    image = image * (0.5 + 0.5 * mask[:, :, np.newaxis])
    image = np.clip(image, 0, 1)
    # 将图像转换回 [0, 255] 范围
    image = (image * 255).astype(np.uint8)
    return image

class LightingTransform:
    def __init__(self, light_directions):
        self.light_directions = light_directions

    def __call__(self, image):
        # 随机选择一个光源方向
        light_direction = np.random.choice(self.light_directions)
        image = np.array(image)
        image = simulate_lighting(image, light_direction)
        return Image.fromarray(image)

def get_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train

        # 定义随机选择的变换
    transform_ae = transforms.RandomChoice([
        transforms.ColorJitter(brightness=0.2),
        transforms.ColorJitter(contrast=0.2),
        transforms.ColorJitter(saturation=0.2),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    ])

    # 定义完整的数据预处理和增强流程
    # data_transforms = transforms.Compose([
    #     transforms.Resize((size, size)),  # 调整图像大小
    #     transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转，概率为 0.5
    #     transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转，概率为 0.5
    #     transforms.RandomRotation(degrees=15),  # 随机旋转，角度范围为 [-15, 15]
    #     transforms.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),  # 随机缩放裁剪
    #     transform_ae,  # 应用颜色抖动增强
    #     transforms.ToTensor(),  # 转换为 Tensor
    #     transforms.CenterCrop(isize),  # 中心裁剪
    #     transforms.Normalize(mean=mean_train, std=std_train)  # 归一化
    # ])

    # 定义数据增强流程
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transform_ae,
        # LightingTransform(light_directions=[0, 90, 180, 270]),  # 模拟 0, 90, 180, 270 度的光源
        transforms.RandomResizedCrop(size, scale=(0.9, 1.2), ratio=(0.9, 1.1)),  # 随机缩放裁剪
        transforms.RandomHorizontalFlip(p=0.25), 
        transforms.RandomVerticalFlip(p=0.25),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train, std=std_train)
    ])

    # data_transforms = transforms.Compose([
    #     transforms.Resize((size, size)),
    #     transform_ae,                                            #TODO 
    #     transforms.ToTensor(),
    #     transforms.CenterCrop(isize),
    #     transforms.Normalize(mean=mean_train,
    #                          std=std_train)])
    



    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    
    infer_transforms = transforms.Compose([
        transforms.Resize((size, size)),                              #TODO 
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    
    return data_transforms, gt_transforms, infer_transforms


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.cls_idx = 0

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class MVTecDataset2(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root,  'test_public')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.cls_idx = 0

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'ground_truth':
                continue
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                gt_paths = glob.glob(os.path.join(self.img_path, 'ground_truth', defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path



class MVTecDataset2_test(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):

        assert phase in {
            'train',
            'validation',
            'test_public',
            'test_private',
            'test_private_mixed',
        }, f'unknown split: {phase}'

        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root,  phase)

        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        if phase in {'train', 'validation', 'test_public'}:
            self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        else:
            self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset_private()  # self.labels => good : 0, anomaly : 1
        self.cls_idx = 0

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'ground_truth':
                continue
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                gt_paths = glob.glob(os.path.join(self.img_path, 'ground_truth', defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)
    
    def load_dataset_private(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        # defect_types = os.listdir(self.img_path)

        img_paths = glob.glob(self.img_path + "/*.png") + \
                            glob.glob(self.img_path + "/*.JPG") + \
                            glob.glob(self.img_path + "/*.bmp")
        img_tot_paths.extend(img_paths)
        gt_tot_paths.extend([0] * len(img_paths))
        tot_labels.extend([0] * len(img_paths))
        tot_types.extend(['good'] * len(img_paths))


        # for defect_type in defect_types:
        #     if defect_type == 'ground_truth':
        #         continue
        #     if defect_type == 'good':
        #         img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
        #                     glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
        #                     glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
        #         img_tot_paths.extend(img_paths)
        #         gt_tot_paths.extend([0] * len(img_paths))
        #         tot_labels.extend([0] * len(img_paths))
        #         tot_types.extend(['good'] * len(img_paths))
        #     else:
        #         img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
        #                     glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
        #                     glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
        #         gt_paths = glob.glob(os.path.join(self.img_path, 'ground_truth', defect_type) + "/*.png")
        #         img_paths.sort()
        #         gt_paths.sort()
        #         img_tot_paths.extend(img_paths)
        #         gt_tot_paths.extend(gt_paths)
        #         tot_labels.extend([1] * len(img_paths))
        #         tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img_shape = img.size
        img = self.transform(img)
        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path, img_shape




class RealIADDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, transform, gt_transform, phase):
        self.img_path = os.path.join(root, 'realiad_1024', category)
        self.transform = transform
        self.gt_transform = gt_transform
        self.phase = phase

        json_path = os.path.join(root, 'realiad_jsons', 'realiad_jsons', category + '.json')
        with open(json_path) as file:
            class_json = file.read()
        class_json = json.loads(class_json)

        self.img_paths, self.gt_paths, self.labels, self.types = [], [], [], []

        data_set = class_json[phase]
        for sample in data_set:
            self.img_paths.append(os.path.join(root, 'realiad_1024', category, sample['image_path']))
            label = sample['anomaly_class'] != 'OK'
            if label:
                self.gt_paths.append(os.path.join(root, 'realiad_1024', category, sample['mask_path']))
            else:
                self.gt_paths.append(None)
            self.labels.append(label)
            self.types.append(sample['anomaly_class'])

        self.img_paths = np.array(self.img_paths)
        self.gt_paths = np.array(self.gt_paths)
        self.labels = np.array(self.labels)
        self.types = np.array(self.types)
        self.cls_idx = 0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        if self.phase == 'train':
            return img, label

        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path



