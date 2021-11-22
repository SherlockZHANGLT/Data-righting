import matplotlib.image as img
import numpy as np
from PIL import Image
import os
def read_directory(di_name):
    for filename in os.listdir(di_name):
        name=filename.split('.')[0]
        input_img=Image.open(di_name+"/"+filename)
        temp=input_img.resize((320,320), Image.ANTIALIAS)
        temp.save("D:/2021/pic/temp"+"/"+name+"_temp.png")
        temp1=img.imread("D:/2021/pic/temp"+"/"+name+"_temp.png")
        out=temp1
        for i in range(320):
           for j in range(320):
                center=np.array([160,160])
                t=np.array([i,j])
                if(sum((t-center)**2))**(1/2)<160:
                   out[i,j,3]=1.0
                else:
                    out[i,j,0]=1.0
                    out[i,j,1]=1.0
                    out[i,j,2]=1.0
        img.imsave("D:/2021/pic/train"+"/"+name+"_cir.png",out)
read_directory("source")