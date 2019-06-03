import os
import cv2
from sklearn.cluster import KMeans
import numpy as np

frame_src_path = "datasets/frames" #图片读取路径
frame_save_path = "datasets/trainA" #图片保存路径
if not os.path.exists(frame_save_path):
    os.mkdir(frame_save_path) #创建文件夹

frames = os.listdir(frame_src_path)

X=[]
for i in range(len(frames)):
    print("---> 正在处理第%d张图片:" % index)
    index = index + 1
    img1 = cv2.imread(frame_src_path + '/' + str(i) + '.jpg')
    img1_new = cv2.resize(img1, (8, 8), interpolation=cv2.INTER_AREA)
    img1_new = cv2.cvtColor(img1_new, cv2.COLOR_BGR2GRAY)
    # 灰度化处理
    img = np.array(img1_new)
    img = img.reshape(-1).tolist()
    X.append(img)
#print(X)
kmeans=KMeans(n_clusters=10)
kmeans.fit(X)
Y=[i for i in range(len(kmeans.cluster_centers_))]
#print(kmeans.cluster_centers_)
for i in range(len(kmeans.cluster_centers_)):
    m=99999
    for j in range(len(X)):
        img = np.array(X[j])
        t = np.sqrt(np.sum(np.power(img-kmeans.cluster_centers_[i],2)))
        if t<m:
            m=t
            Y[i]=j

k = 10 #聚类 k 取值
frame_src_path = "datasets/frames"
frame_save_path = "datasets/keyframes"
os.mkdir(frame_save_path)#创建关键帧目录
frames_1 = os.listdir(frame_src_path)#进入目录

X = np.zeros([1,720*480])
for i in range(len(frames_1)):
    print("正在读取第",i,"张图片---：")
    img = cv2.imread(frame_src_path + '/' + str(i) + '.jpg')  # 读取图片
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # RGB灰度化处理
    img = np.array(img)
    img = img.ravel()
    if i==0:
        X = img
    else:
        X = np.vstack((X,img))
print("正在进行聚类----：")
y = KMeans(n_clusters = k).fit(X)
Y=[i for i in range(len(y.cluster_centers_))]
print("寻找中心点----：")
for i in range(len(y.cluster_centers_)):
    D = 999999
    for j in range(len(y.labels_)):
        d = np.sqrt(np.sum(np.power(X[j]-y.cluster_centers_[i],2)))
        if d < D:
            D = d
            #中心点数组保存中心点的图片编号
            Y[i] = j
Y.sort()
print(Y)#打印一下图片编号数组
#图片保存：
for i in range(len(Y)):
    print("正在保存第",i,"张图片----：")
    img = cv2.imread(frame_src_path + '/' + str(Y[i]) + '.jpg')
    cv2.imwrite(frame_save_path + '/' + "%d.jpg" % i, img)


