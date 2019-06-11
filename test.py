# 导入需要的库
import skimage.io
import tensorflow as tf
import time
# 导入自定义的函数
from model import *
from get_data import *
to_test = True  # 不进行test
to_restore = False  # 设置为true恢复模型
output_path = "./output"  # 设置输出文件路径
check_dir = "./output/checkpoints/"  # 模型参数的保存路径
data_dir = "./datasets"  # 数据的根目录

save_training_images = True

# 定义训练过程
def test():
    # 读取数据
    data_A, _ = get_data(data_dir, "/test", "/trainB")
    with tf.variable_scope('Model'):
        # 输入数据的占位符
        input_A = tf.placeholder(tf.float32, [batch_size, image_height, image_width, image_channel], name="input_A")
        # 建立生成器
        fake_B = build_generator_resnet_9blocks(input_A, "g_A")  # 输入A生成A’
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 设定显存不超量使用
    with tf.Session() as sess:  # 建立会话层
        # 结果保存器
        model_vars = tf.trainable_variables()
        g_A_vars = [var for var in model_vars if 'g_A' in var.name]
        #print(g_A_vars)
        saver = tf.train.Saver(var_list=g_A_vars)
        checkpoint = tf.train.latest_checkpoint(check_dir)
        #print(checkpoint)
        saver.restore(sess, checkpoint)

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for i in range(len(data_A)):
            print("正在处理第%d张图片" %(i))
            fake = sess.run([fake_B],feed_dict={input_A: np.reshape(data_A[i], [-1, 256, 256, 3])})
            #print(fake[0].shape)
            if (save_training_images):
                # 检查路径是否存在
                if not os.path.exists("./output/res"):
                    os.makedirs("./output/res")
                    # 保存10张影像
                skimage.io.imsave("./output/res/fake_" + str(i) + ".jpg",
                                  np.reshape(((fake[0] + 1) * 127.5).astype(np.uint8), [256, 256, 3]))
                # 保存图像结束------------------------------------------------------------
