import os
import cv2

videos_src_path = "datasets/mp4"
videoframe_save_path = "datasets/frames"
width = 720 #宽
height = 480 #高
time_interval = 2 #隔几帧取一次


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
        print("正在读取视频：", each_video)
        if not os.path.exists(frame_save_path):
            os.mkdir(frame_save_path)
        each_video_full_path = os.path.join(video_src_path, each_video)

        cap = cv2.VideoCapture(each_video_full_path)
        # 保存视频里所有图片信息
        frame_index = 0
        frame_count = 0
        if cap.isOpened():
            success = True
        else:
            success = False
            print("读取失败!")

        while(success):
            success, frame = cap.read()
            print("---> 正在读取第%d帧:" % frame_index, success)
            if frame_index % interval == 0 and success:     # 如路径下有多个视频文件时视频最后一帧报错因此条件语句中加and success
                resize_frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
                img = resize_frame[60:420,0:720]
                cv2.imwrite(frame_save_path + '/' + "%d.jpg" % frame_count, img)
                frame_count += 1

            frame_index += 1
    cap.release()



