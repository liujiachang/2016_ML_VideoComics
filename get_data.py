import os
import numpy as np
from PIL import Image
from keyframes_3 import *

image_width = 256 #宽
image_height = 256 #高
image_channel = 3 #通道数

# 读取数据到内存当中
def get_data(input_dir, floderA, floderB):
    '''
    函数功能：输入根路径，和不同数据的文件夹，读取数据
    :param input_dir:根目录的参数
    :param floderA: 数据集A所在的文件夹名
    :param floderB: 数据集B所在的文件夹名
    :return: 返回读取好的数据，train_set_A即A文件夹的数据, train_set_B即B文件夹的数据
    '''
    if not os.path.exists(input_dir + floderA):
        fun()
    # 读取路径，并判断路径下有多少张影像
    print("读取路径，并判断路径下有多少张影像")
    imagesA = os.listdir(input_dir + floderA)
    imagesB = os.listdir(input_dir + floderB)
    imageA_len = len(imagesA)
    imageB_len = len(imagesB)

    # 定义用于存放读取影像的变量
    print("定义用于存放读取影像的变量")
    dataA = np.empty((imageA_len, image_height, image_width, image_channel), dtype="float32")
    dataB = np.empty((imageB_len, image_height, image_width, image_channel), dtype="float32")
    #print(dataA)
    #print(dataB)

    # 读取文件夹A中的数据
    print("读取文件夹A中的数据")
    for i in range(imageA_len):
        # 逐个影像读取
        img = Image.open(input_dir + floderA + "/" + imagesA[i])
        #img = img.resize((image_width, image_height))
        arr = np.asarray(img, dtype="float32")
        # 对影像数据进行归一化[-1, 1]，并将结果保存到变量中
        dataA[i, :, :, :] = arr * 1.0 / 127.5 - 1.0
        #print(arr.shape)
    print(dataA.shape)
    # 读取文件夹B中的数据
    print("读取文件夹B中的数据")
    for i in range(imageB_len):
        # 逐个影像读取
        img = Image.open(input_dir + floderB + "/" + imagesB[i])
        #img = img.resize((image_width, image_height))
        arr = np.asarray(img, dtype="float32")
        # 对影像数据进行归一化[-1, 1]，并将结果保存到变量中
        dataB[i, :, :, :] = arr * 1.0 / 127.5 - 1.0
    print(dataB.shape)
    return dataA, dataB

#src = "datasets/"
#trainA = "trainA"
#trainB = "trainB"
#A, B = get_data(src,trainA,trainB)
#print(A)
#print(B)