# 导入需要的库
import skimage.io
import tensorflow as tf
import time
# 导入model中的函数
from model import *
from get_data import *

to_train = False
to_restore = False  # 设置为true继续训练
output_path = "./output"  # 设置输出文件路径
check_dir = "./output/checkpoints/"  # 输出模型参数的文件路径
data_dir = "./datasets"  # 数据的根目录

epochs = 5000
max_images = 10


save_training_images = True


# 定义训练过程
def train():
    # 读取数据
    data_A, data_B = get_data(data_dir, "/trainA", "/trainB")

    # CycleGAN的模型构建 ----------------------------------------------------------
    # 输入数据的占位符
    input_A = tf.placeholder(tf.float32, [batch_size, image_height, image_width, image_channel], name="input_A")
    input_B = tf.placeholder(tf.float32, [batch_size, image_height, image_width, image_channel], name="input_B")

    fake_pool_A = tf.placeholder(tf.float32, [None, image_height, image_width, image_channel], name="fake_pool_A")
    fake_pool_B = tf.placeholder(tf.float32, [None, image_height, image_width, image_channel], name="fake_pool_B")

    global_step = tf.Variable(0, name="global_step", trainable=False)

    num_fake_inputs = 0

    lr = tf.placeholder(tf.float32, shape=[], name="lr")

    # 建立生成器和判别器
    with tf.variable_scope("Model") as scope:

        fake_B = build_generator_resnet_9blocks(input_A, "g_A")  # 输入A生成A’
        fake_A = build_generator_resnet_9blocks(input_B, "g_B")  # 输入B生成B‘
        rec_A = build_gen_discriminator(input_A, "d_A")# 判断A是不是真
        rec_B = build_gen_discriminator(input_B, "d_B")# 判断B是不是真

        scope.reuse_variables()

        fake_rec_A = build_gen_discriminator(fake_A, "d_A")  # 判断B‘是不是真
        fake_rec_B = build_gen_discriminator(fake_B, "d_B")  # 判断A‘是不是真
        cyc_A = build_generator_resnet_9blocks(fake_B, "g_B")# 将A’还原成A''
        cyc_B = build_generator_resnet_9blocks(fake_A, "g_A")# 将B’还原成B''

        scope.reuse_variables()

        fake_pool_rec_A = build_gen_discriminator(fake_pool_A, "d_A")
        fake_pool_rec_B = build_gen_discriminator(fake_pool_B, "d_B")

    # 定义损失函数
    cyc_loss = tf.reduce_mean(tf.abs(input_B - cyc_B)) + tf.reduce_mean(tf.abs(input_A - cyc_A))

    disc_loss_A = tf.reduce_mean(tf.squared_difference(fake_rec_A, 1))
    disc_loss_B = tf.reduce_mean(tf.squared_difference(fake_rec_B, 1))
    n = 10.0
    g_loss_A = cyc_loss * n + disc_loss_A
    g_loss_B = cyc_loss * n + disc_loss_B

    d_loss_A = (tf.reduce_mean(tf.square(fake_pool_rec_A)) + tf.reduce_mean(
        tf.squared_difference(rec_A, 1))) / 2.0
    d_loss_B = (tf.reduce_mean(tf.square(fake_pool_rec_B)) + tf.reduce_mean(
        tf.squared_difference(rec_B, 1))) / 2.0

    # 定义优化器
    optimizer = tf.train.AdamOptimizer(lr, beta1 = 0.5)

    model_vars = tf.trainable_variables()

    d_A_vars = [var for var in model_vars if 'd_A' in var.name]
    g_A_vars = [var for var in model_vars if 'g_A' in var.name]
    d_B_vars = [var for var in model_vars if 'd_B' in var.name]
    g_B_vars = [var for var in model_vars if 'g_B' in var.name]

    d_A_trainer = optimizer.minimize(d_loss_A, var_list=d_A_vars)
    d_B_trainer = optimizer.minimize(d_loss_B, var_list=d_B_vars)
    g_A_trainer = optimizer.minimize(g_loss_A, var_list=g_A_vars)
    g_B_trainer = optimizer.minimize(g_loss_B, var_list=g_B_vars)

    for var in model_vars:
        print(var.name)

    # Summary variables for tensorboard

    g_A_loss_summ = tf.summary.scalar("g_A_loss", g_loss_A)
    g_B_loss_summ = tf.summary.scalar("g_B_loss", g_loss_B)
    d_A_loss_summ = tf.summary.scalar("d_A_loss", d_loss_A)
    d_B_loss_summ = tf.summary.scalar("d_B_loss", d_loss_B)
    # 模型构建完毕-------------------------------------------------------------------

    # 生成结果的存储器
    fake_images_A = np.zeros((pool_size, 1, image_height, image_width, image_channel))
    fake_images_B = np.zeros((pool_size, 1, image_height, image_width, image_channel))

    # 全局变量初始化
    init = tf.global_variables_initializer()
    # 结果保存器
    saver = tf.train.Saver()
    # 恢复模型
    if to_restore:
        remodel_d_A = tf.train.Saver(var_list=d_A_vars)
        remodel_d_B = tf.train.Saver(var_list=d_B_vars)
        remodel_g_A = tf.train.Saver(var_list=g_A_vars)
        remodel_g_B = tf.train.Saver(var_list=g_B_vars)
    with tf.Session() as sess:
        sess.run(init)
        if to_restore:
            ckpt = tf.train.get_checkpoint_state(check_dir)
            remodel_d_A.restore(sess, ckpt.model_checkpoint_path)
            remodel_g_A.restore(sess, ckpt.model_checkpoint_path)
            remodel_d_B.restore(sess, ckpt.model_checkpoint_path)
            remodel_g_B.restore(sess, ckpt.model_checkpoint_path)
        writer = tf.summary.FileWriter("./output/2")

        if not os.path.exists(check_dir):
            os.makedirs(check_dir)

        # 开始训练
        for epoch in range(sess.run(global_step), epochs):
            n = n - epoch / 100.0
            t = time.time()
            np.random.shuffle(data_B)
            np.random.shuffle(data_A)
            #保存模型
            saver.save(sess, os.path.join(check_dir, "cyclegan"), global_step=epoch)

            # 按照训练的epoch调整学习率。
            if (epoch < 100):
                curr_lr = 0.0002
            elif (epoch < 200):
                curr_lr = 0.0002 - 0.0002*(epoch-100)/100
            else:
                curr_lr = 0.0
            # 保存图像-----------------------------------------------------------------
            if (save_training_images):
                # 检查路径是否存在
                if not os.path.exists("./output/imgs"):
                    os.makedirs("./output/imgs")

                # 保存10张影像
                for i in range(0, 10):
                    fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run(
                        [fake_A, fake_B, cyc_A, cyc_B],
                        feed_dict={input_A: np.reshape(data_A[i], [-1, 256, 256, 3]),
                                   input_B: np.reshape(data_B[i], [-1, 256, 256, 3])})
                    # fake表示输入A，通过B的特征而变成B
                    #skimage.io.imsave("./output/imgs/fakeB_" + str(epoch) + "_" + str(i) + ".jpg",
                           #((fake_A_temp[0] + 1) * 127.5).astype(np.uint8))
                    skimage.io.imsave("./output/imgs/fake_" + str(epoch) + "_" + str(i) + ".jpg",
                           ((fake_B_temp[0] + 1) * 127.5).astype(np.uint8))
                    #skimage.io.imsave("./output/imgs/real_" + str(epoch) + "_" + str(i) + ".jpg",
                            #((data_A[i] + 1) * 127.5).astype(np.uint8))
                    # cyc表示输入A，通过B的特征变成B，再由A的特征变成A结果
                    #skimage.io.imsave("./output/imgs/cycA_" + str(epoch) + "_" + str(i) + ".jpg",
                           #((cyc_A_temp[0] + 1) * 127.5).astype(np.uint8))
                    #skimage.io.imsave("./output/imgs/cycB_" + str(epoch) + "_" + str(i) + ".jpg",
                           #((cyc_B_temp[0] + 1) * 127.5).astype(np.uint8))

            # 保存图像结束------------------------------------------------------------

            # 循环执行cycleGAN
            for ptr in range(0, max_images):
                #print("In the iteration ", ptr)
                # Optimizing the G_A network
                _, fake_B_temp, G_A_loss, summary_str = sess.run([g_A_trainer, fake_B, g_loss_A, g_A_loss_summ],
                                                       feed_dict={input_A: np.reshape(data_A[ptr], [-1, 256, 256, 3]),
                                                                  input_B: np.reshape(data_B[ptr], [-1, 256, 256, 3]),
                                                                  lr: curr_lr})

                writer.add_summary(summary_str, epoch * max_images + ptr)

                fake_B_temp1 = fake_image_pool(num_fake_inputs, fake_B_temp, fake_images_B)

                # Optimizing the D_B network
                if epoch % 2 == 0:
                    _, D_B_loss, summary_str = sess.run([d_B_trainer, d_loss_B, d_B_loss_summ],
                                          feed_dict={input_A: np.reshape(data_A[ptr], [-1, 256, 256, 3]),
                                                     input_B: np.reshape(data_B[ptr], [-1, 256, 256, 3]),
                                                     lr: curr_lr,
                                                     fake_pool_B: fake_B_temp1})
                writer.add_summary(summary_str, epoch * max_images + ptr)

                # Optimizing the G_B network
                _, fake_A_temp, G_B_loss, summary_str = sess.run([g_B_trainer, fake_A, g_loss_B, g_B_loss_summ],
                                                       feed_dict={input_A: np.reshape(data_A[ptr], [-1, 256, 256, 3]),
                                                                  input_B: np.reshape(data_B[ptr], [-1, 256, 256, 3]),
                                                                  lr: curr_lr})

                writer.add_summary(summary_str, epoch * max_images + ptr)

                fake_A_temp1 = fake_image_pool(num_fake_inputs, fake_A_temp, fake_images_A)

                # Optimizing the D_A network
                if epoch % 2 == 0:
                    _, D_A_loss, summary_str = sess.run([d_A_trainer, d_loss_A, d_A_loss_summ],
                                          feed_dict={input_A: np.reshape(data_A[ptr], [-1, 256, 256, 3]),
                                                     input_B: np.reshape(data_B[ptr], [-1, 256, 256, 3]),
                                                     lr: curr_lr,
                                                     fake_pool_A: fake_A_temp1})

                writer.add_summary(summary_str, epoch * max_images + ptr)

                num_fake_inputs += 1
            print("In the epoch = %d, time = %f, G_A_loss = %f, D_A_loss = %f, G_B_loss = %f, D_B_loss = %f" %(epoch, time.time()-t, G_A_loss, D_A_loss, G_B_loss, D_B_loss))
            sess.run(tf.assign(global_step, epoch + 1))

        writer.add_graph(sess.graph)