import os
import cv2

videos_src_path = "datasets/mp4"
frame_save_path = "datasets/frames"
width = 720
height = 480
time_interval = 1


def video2frame(video_src_path, frame_save_path, frame_width, frame_height, interval):
    """
    将视频按固定间隔读取写入图片
    :param video_src_path: 视频存放路径
    :param formats:　包含的所有视频格式
    :param frame_save_path:　保存路径
    :param frame_width:　保存帧宽
    :param frame_height:　保存帧高
    :param interval:　保存帧间隔
    :return:　帧图片
    """
    videos = os.listdir(video_src_path)

    for each_video in videos:
        # print "正在读取视频：", each_video
        print("正在读取视频：", each_video)    # 我的是Python3.6

        os.mkdir(frame_save_path)
        each_video_full_path = os.path.join(video_src_path, each_video)

        cap = cv2.VideoCapture(each_video_full_path)
        frame_index = 0
        frame_count = 0
        if cap.isOpened():
            success = True
        else:
            success = False
            print("读取失败!")

        while(success):
            success, frame = cap.read()
            print("---> 正在读取第%d帧:" % frame_index, success)      # 我的是Python3.6

            if frame_index % interval == 0 and success:     # 如路径下有多个视频文件时视频最后一帧报错因此条件语句中加and success
                resize_frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
                img = resize_frame[60:420,0:720]
                cv2.imwrite(frame_save_path + '/' + "%d.jpg" % frame_count, img)
                frame_count += 1

            frame_index += 1

    cap.release()		# 这行要缩一下、原博客会报错(全局变量与局部变量)

def main():
    video2frame(videos_src_path, frame_save_path, width, height, time_interval)
if __name__ == '__main__':
    main()


