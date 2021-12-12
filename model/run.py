# import区域，sys为必须导入，其他根据需求导入
import os
import sys
import random
import numpy as np
import pandas as pd
import cv2
import torch
from model import ResNet as MyNet
from glob import glob
# 主函数，格式固定，to_pred_dir为预测所在文件夹，result_save_path为预测结果生成路径
# 以下为示例
def main(to_pred_dir, result_save_path):
    runpyp = os.path.abspath(__file__)
    modeldirp = os.path.dirname(runpyp)
    modelp = os.path.join(modeldirp,"model.pth")

    mynet = MyNet().cuda()
    mynet = torch.nn.DataParallel(mynet)
    mynet.load_state_dict(torch.load(modelp))
    mynet.eval()

    pred_imgs = os.listdir(to_pred_dir)
    pred_imgsp_lines = [os.path.join(to_pred_dir,p) for p in pred_imgs]
    imgs_id = [os.path.splitext(p)[0] for p in pred_imgs]
    pred_list = []
    for imgpath in pred_imgsp_lines:
        print(imgpath)
        img = cv2.imread(imgpath)
        img = cv2.resize(img,(224,224))
        img = img/255
        img = torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0).cuda()
        pred = mynet(img)
        pred = pred.detach()
        pred_label = torch.argmax(pred,dim = 1).cpu().item()
        pred_list.append(pred_label)
    preds = np.array(pred_list)    
    df = pd.DataFrame({"id":imgs_id,"label":preds})
    df.to_csv(result_save_path,index=None)

# ！！！注意：
# 图片赛题给出的参数为to_pred_dir,是一个文件夹，其图片内容为
# to_pred_dir/to_pred_0.png
# to_pred_dir/to_pred_1.png
# to_pred_dir/......
# 所需要生成的csv文件头为id,label,如下
# image_id,label
# to_pred_0,4
# to_pred_1,76
# to_pred_2,...

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    to_pred_dir = sys.argv[1]  # 所需预测的文件夹路径
    result_save_path = sys.argv[2]  # 预测结果保存文件路径
    main(to_pred_dir, result_save_path)
