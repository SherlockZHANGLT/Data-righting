import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms   
import cv2
import os
import albumentations as A
from PIL import Image
import random
import numpy as np
trans = A.Compose([
        A.GaussNoise(p=0.3),    # 将高斯噪声应用于输入图像。
        A.Cutout(num_holes=5, max_h_size=3, max_w_size=3, fill_value=255, p=0.3),
    ])
class Mydataset(Dataset):
    def __init__(self, data_dir_rig = './train/旋转后图片/', data_dir_raw = './train/原始图片/',train = False):
        imgsname = []
        imgspath_rig = []
        imgspath_raw = []
        imgsclass = []
        for k,i in enumerate(os.listdir(data_dir_rig)):
            for j in os.listdir(data_dir_rig + i):
                imgsname.append(j)
                imgsclass.append(k)
                imgspath_rig.append(os.path.join(data_dir_rig + i, j))
                imgname_raw =  "_".join(j.split("_")[:2])+".png"
                imgspath_raw.append(os.path.join(data_dir_raw + i, imgname_raw))
        
        self.train = train
        self.imgsname = imgsname
        self.imgspath_rig = imgspath_rig
        self.imgspath_raw = imgspath_raw
        self.imgsclass = imgsclass
        print("Train Images Num:",len(self.imgsname))
        
    def __len__(self):
        return len(self.imgsname)
    
    def __getitem__(self, index):
        img_name = self.imgsname[index]
        img_path_rig = self.imgspath_rig[index]
        img_path_raw = self.imgspath_raw[index]
        image_class = self.imgsclass[index]
        label = img_name.split("_")[-1].split(".")[0][1:]
        
        image_rig = cv2.imread(img_path_rig)
        image_raw = cv2.imread(img_path_raw)
        image_rig = cv2.resize(image_rig,(224,224))
        image_raw = cv2.resize(image_raw,(224,224))        
        if self.train:
            agu = trans(image = image_rig)
            image_rig = agu['image']

            
            # RandomRotation
            if random.random()<0.8:
                image_rig = Image.fromarray(image_rig)
                rotnum = random.randint(0,359)
                image_rig = transforms.functional.rotate(image_rig, rotnum)
                label = (int(label) + rotnum)%360
                image_rig = np.array(image_rig)

        image_rig = image_rig/255
        image_raw = image_raw/255

        image_rig = torch.from_numpy(image_rig).float().permute(2,0,1)
        image_raw = torch.from_numpy(image_raw).float().permute(2,0,1)
        label = torch.Tensor([int(label)]).long()
        image_class = torch.Tensor([int(image_class)]).long()
        
        return image_rig, image_raw, label, image_class
class Mydataset_extra(Dataset):
    def __init__(self, data_dir_rig = './train/原图片/',train = False):
        imgsname = []
        imgspath_rig = []
        imgspath_raw = []
        imgsclass = []
        for i in os.listdir(data_dir_rig):
            for j in os.listdir(data_dir_rig + i):
                imgsname.append(j)
                imgsclass.append(i)
                imgspath_rig.append(os.path.join(data_dir_rig + i, j))
        
        self.train = train
        self.imgsname = imgsname
        self.imgspath_rig = imgspath_rig
        self.imgsclass = imgsclass
        print("Extra Train Images Num:",len(self.imgsname))
        
    def __len__(self):
        return len(self.imgsname)
    
    def __getitem__(self, index):
        img_name = self.imgsname[index]
        img_path_rig = self.imgspath_rig[index]
        img_class = self.imgsclass[index]
        label = img_name.split("_")[-1].split(".")[0][1:]
        
        image_rig = cv2.imread(img_path_rig)
        image_raw = image_rig
        image_rig = cv2.resize(image_rig,(224,224))
        image_raw = cv2.resize(image_raw,(224,224))        
        if self.train:
            agu = trans(image = image_rig)
            image_rig = agu['image']

            
            # RandomRotation
            image_rig = Image.fromarray(image_rig)
            rotnum = random.randint(0,359)
            image_rig = transforms.functional.rotate(image_rig, rotnum)
            label = (int(label) + rotnum)%360
            image_rig = np.array(image_rig)

        image_rig = image_rig/255
        image_raw = image_raw/255

        image_rig = torch.from_numpy(image_rig).float().permute(2,0,1)
        image_raw = torch.from_numpy(image_raw).float().permute(2,0,1)
        label = torch.Tensor([int(label)]).long()
        # img_class = torch.Tensor(int(img_class))
        
        return image_rig, image_raw, label#, img_class

if __name__ == '__main__':
    DataSet = Mydataset()