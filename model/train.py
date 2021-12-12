import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import random
import os
import sys
import argparse
import time
from Mydataset import Mydataset,Mydataset_extra
from model import ResNet as MyNet
# from res2net import se_res2net50 as MyNet
# from swin_transformer import SwinTransformer as MyNet
# from densenet import densenet121 as MyNet
import numpy as np
from torchvision import transforms
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in')
        nn.init.constant_(m.bias, 0.0)
def getHeatMap(label,class_num = 360,A=1.0,sigmas=3):
    label = label.unsqueeze(1)
    x_range = torch.arange(0,class_num)
    x_range = x_range.unsqueeze(0)
    x_range= x_range.expand(label.shape[0],class_num).cuda()
    target = label.expand(label.shape[0],class_num).cuda()
    sigmas=torch.ones((label.shape[0],class_num))*sigmas
    sigmas=sigmas.cuda()

    squr_distance=torch.pow(target-x_range,2)    #计算各坐标与指定坐标的距离平方
    heatmap=torch.exp(-squr_distance/(2*torch.pow(sigmas,2))).float()    #生成以x0，y0为中心，半径为sigmas，heatmap_size大小的高斯分布热力图
    return heatmap


def train(config):

    os.environ['CUDA_VISIBLE_DEVICES']=config.gpu_device

    mynet = MyNet().cuda()
    mynet = torch.nn.DataParallel(mynet)
    if config.load_pretrain == True:
        mynet.load_state_dict(torch.load(config.pretrain_dir))
    optimizer = torch.optim.Adam(mynet.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    bceloss = nn.BCEWithLogitsLoss(reduction='mean')
    celoss = nn.CrossEntropyLoss(reduction='mean')
    mynet.train()
    train_dataset = Mydataset(config.data_dir_rig,config.data_dir_raw,train = True)	
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    extra_dataset = Mydataset_extra('/data/L_E_Data/ImageRotationDataset/extra/',train = True)
    extra_loader = torch.utils.data.DataLoader(extra_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    sigmas = 3

    for epoch in range(config.num_epochs):
        dis_list = []
        acc_list = []
        if epoch in config.change_lr_rpochlist:
            config.lr = 0.5*config.lr
            optimizer = torch.optim.Adam(mynet.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        for iteration, (image_rig, image_raw, label) in enumerate(extra_loader):
            image_rig = image_rig.cuda()
            label = label.cuda()
            label = label.squeeze()
            predict = mynet(image_rig)
            pre = torch.argmax(predict,dim = 1)
            dis = torch.mean((pre.float()-label.float())**2)
            acc = torch.sum((pre==label).int())/pre.shape[0]
            label_heatmap = getHeatMap(label,sigmas=sigmas)
            loss = bceloss(predict,label_heatmap)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            dis_list.append(dis.item())
            acc_list.append(acc.item())
        torch.save(mynet.state_dict(), config.snapshots_folder + "model" + config.train_mode + '.pth') 
        print('Epoch:',epoch+1,' Loss: ',loss.item(),'acc',np.mean(acc_list),'dis',np.mean(dis_list))

        for iteration, (image_rig, image_raw, label,image_class) in enumerate(train_loader):
            image_rig = image_rig.cuda()
            label = label.cuda()
            label = label.squeeze()
            predict = mynet(image_rig)
            pre = torch.argmax(predict,dim = 1)
            dis = torch.mean((pre.float()-label.float())**2)
            acc = torch.sum((pre==label).int())/pre.shape[0]
            label_heatmap = getHeatMap(label,sigmas=sigmas)
            loss = bceloss(predict,label_heatmap)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            dis_list.append(dis.item())
            acc_list.append(acc.item())
        torch.save(mynet.state_dict(), config.snapshots_folder + "model" + config.train_mode + '.pth') 
        print('Epoch:',epoch+1,' Loss: ',loss.item(),'acc',np.mean(acc_list),'dis',np.mean(dis_list))


    # for epoch in range(config.num_epochs):
    #     dis_list = []
    #     acc_list = []
    #     if epoch in config.change_lr_rpochlist:
    #         config.lr = 0.5*config.lr
    #         optimizer = torch.optim.Adam(mynet.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    #     for iteration, (image_rig, image_raw, label) in enumerate(extra_loader):
    #         image_rig = image_rig.cuda()
    #         label = label.cuda()
    #         label = label.squeeze()
    #         predict = mynet(image_rig)
    #         pre = torch.argmax(predict,dim = 1)
    #         dis = torch.mean((pre.float()-label.float())**2)
    #         acc = torch.sum((pre==label).int())/pre.shape[0]
    #         label_heatmap = getHeatMap(label,sigmas=sigmas)
    #         loss = bceloss(predict,label_heatmap)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         dis_list.append(dis.item())
    #         acc_list.append(acc.item())
    #     torch.save(mynet.state_dict(), config.snapshots_folder + "model" + config.train_mode + '.pth') 
    #     print('Epoch:',epoch+1,' Loss: ',loss.item(),'acc',np.mean(acc_list),'dis',np.mean(dis_list))

    for epoch in range(config.num_epochs):
        dis_list = []
        acc_list = []
        if epoch in config.change_lr_rpochlist:
            config.lr = 0.5*config.lr
            optimizer = torch.optim.Adam(mynet.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        for iteration, (image_rig, image_raw, label,image_class) in enumerate(train_loader):
            image_rig = image_rig.cuda()
            label = label.cuda()
            label = label.squeeze()
            predict = mynet(image_rig)
            pre = torch.argmax(predict,dim = 1)
            dis = torch.mean((pre.float()-label.float())**2)
            acc = torch.sum((pre==label).int())/pre.shape[0]
            label_heatmap = getHeatMap(label,sigmas=sigmas)
            loss = bceloss(predict,label_heatmap)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            dis_list.append(dis.item())
            acc_list.append(acc.item())
        torch.save(mynet.state_dict(), config.snapshots_folder + "model" + config.train_mode + '.pth') 
        print('Epoch:',epoch+1,' Loss: ',loss.item(),'acc',np.mean(acc_list),'dis',np.mean(dis_list))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--gpu_device', type=str, default='3')
    parser.add_argument('--train_mode', type=str, default='')
    parser.add_argument('--num_epochs', type= int, default=400)
    parser.add_argument('--change_lr_rpochlist', type= list, default=[100,150,200,250,300,350])
    parser.add_argument('--start', type= int, default=0)
    parser.add_argument('--data_dir_rig', type=str, default="/data/L_E_Data/ImageRotationDataset/train/rotated/")
    parser.add_argument('--data_dir_raw', type=str, default="/data/L_E_Data/ImageRotationDataset/train/raw/")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--train_batch_size', type=list, default=64)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--snapshots_folder', type=str, default="./")
    parser.add_argument('--load_pretrain', type=bool, default= False)
    parser.add_argument('--pretrain_dir', type=str, default= "")


    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
		    os.mkdir(config.snapshots_folder)
    train(config)








	
