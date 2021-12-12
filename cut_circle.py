import matplotlib.image as img
import numpy as np
from PIL import Image
import os
def read_directory(di_name):
    n=2000
    for filename in os.listdir(di_name):
        input_img=Image.open(di_name+"/"+filename).convert('RGBA')
        temp=input_img.resize((320,320), Image.ANTIALIAS)
        temp.save("D:/2021/pic/temp"+"/"+str(n)+"_temp.png")
        temp1=img.imread("D:/2021/pic/temp"+"/"+str(n)+"_temp.png")
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
        img.imsave("D:/2021/pic/ttem"+"/"+str(n)+"_cir.png",out)
        output_img=Image.open("D:/2021/pic/ttem"+"/"+str(n)+"_cir.png").convert('RGB')
        output_img.save("D:/2021/pic/train"+"/"+str(n)+"_cir_3_r0.png")
        n=n+1
read_directory("source")
