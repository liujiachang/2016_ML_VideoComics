import cv2
import os
import numpy as np

frame_src_path = "datasets/frames"
frame_save_path = "datasets/trainA"
if not os.path.exists(frame_save_path):
    os.mkdir(frame_save_path)
frames = os.listdir(frame_src_path)

count = 0

for i in range(len(frames)):
    print("---> 正在处理第%d张图片:" % i)
    img1 = cv2.imread(frame_src_path + '/' + str(i) + '.jpg')
    img1_new = cv2.resize(img1, (8, 8), interpolation=cv2.INTER_AREA)
    img1_new = cv2.cvtColor(img1_new, cv2.COLOR_BGR2GRAY)
    # 灰度化处理
    if count == 0:
        img1 = cv2.resize(img1, (256, 256), interpolation=cv2.INTER_AREA)
        cv2.imwrite(frame_save_path + '/' + "%d.jpg" % count, img1)
        count=count+1
    else:
        img2 = cv2.imread(frame_save_path + '/' + str(count-1) + '.jpg')
        img2_new = cv2.resize(img2, (8, 8), interpolation=cv2.INTER_AREA)
        img2_new = cv2.cvtColor(img2_new, cv2.COLOR_BGR2GRAY)
        t1 = np.mean(np.array(img1_new))
        t1_new = np.array(img1_new)-t1
        t2 = np.mean(np.array(img2_new))
        t2_new = np.array(img2_new) - t2
        k=0
        for i in range(8):
            for j in range(8):
                if t1_new[i][j]>=0:
                    t1_new[i][j]=1
                else:
                    t1_new[i][j]=0
                if t2_new[i][j]>0:
                    t2_new[i][j]=1
                else:
                    t2_new[i][j]=0
        for i in range(8):
            for j in range(8):
                if(t1_new[i][j]!=t2_new[i][j]):
                    k=k+1
        #print(k)
        if k>10:
            img1 = cv2.resize(img1, (256, 256), interpolation=cv2.INTER_AREA)
            cv2.imwrite(frame_save_path + '/' + "%d.jpg" % count, img1)
            count = count + 1

