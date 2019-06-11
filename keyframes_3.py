
from sklearn.cluster import KMeans
import numpy as np
from keyframes_1 import *

def fun():
    video2frame(videos_src_path, videoframe_save_path, width, height, time_interval)
    frame_src_path = "datasets/frames" #图片读取路径
    frame_save_path = "datasets/trainA" #图片保存路径
    if not os.path.exists(frame_save_path):
        os.mkdir(frame_save_path) #创建文件夹

    frames = os.listdir(frame_src_path)

    X = np.zeros([len(frames),64])
    for i in range(len(frames)):
        print("---> 正在处理第%d张图片:" % i)
        img1 = cv2.imread(frame_src_path + '/' + str(i) + '.jpg')
        img1_new = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1_new = cv2.resize(img1_new, (8, 8), interpolation=cv2.INTER_AREA)
        # 灰度化处理
        img = np.array(img1_new)
        img = img.reshape(-1)
        X[i] = img

    #print(X)
    k = 100 # 聚类数量
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(X)
    Y=[i for i in range(k)]
    #print(kmeans.cluster_centers_)
    for i in range(k):
        m=99999
        for j in range(len(X)):
            img = np.array(X[j])
            t = np.sqrt(np.sum(np.power(img-kmeans.cluster_centers_[i],2)))
            if t<m:
                m=t
                Y[i]=j

    Y.sort()
    print(Y)#打印一下图片编号数组
    #图片保存：
    for i in range(len(Y)):
        print("正在保存第",i,"张图片----：")
        img = cv2.imread(frame_src_path + '/' + str(Y[i]) + '.jpg')
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        cv2.imwrite(frame_save_path + '/' + "%d.jpg" % i, img)


