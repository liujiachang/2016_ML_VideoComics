from sklearn.cluster import KMeans
import os
import cv2
import numpy as np

frame_src_path = "datasets/frames" #图片读取路径
frame_save_path = "datasets/keyframes" #图片保存路径
if not os.path.exists(frame_save_path):
    os.mkdir(frame_save_path) #创建文件夹

frames_1 = os.listdir(frame_src_path)

count = 0
index = 0
X=[]
for i in range(len(frames_1)):
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
kmeans=KMeans(n_clusters=10,)
kmeans.fit(X)
Y=[i for i in range(len(kmeans.cluster_centers_))]
#print(kmeans.cluster_centers_)
for i in range(len(kmeans.cluster_centers_)):
    m=99999
    for j in range(len(X)):
        img = np.array(X[j])
        t=np.sqrt(np.sum(np.power(img-kmeans.cluster_centers_[i],2)))
        if t<m:
            m=t
            Y[i]=j

Y.sort()
print(Y)
#print(kmeans.labels_)

for i in range(len(Y)):
    img = cv2.imread(frame_src_path + '/' + str(Y[i]) + '.jpg')
    #print(img)
    cv2.imwrite(frame_save_path + '/' + "%d.jpg" % i, img)

