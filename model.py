# 导入编写好的layer层函数文件
from net.layer import *

image_width = 256 #宽
image_height = 256 #高
image_channel = 3 #通道数
image_size = image_height * image_width #图像大小

batch_size = 1 #批次

ngf = 32
ndf = 64


# 定义resnet层 残差网络
def build_resnet_block(inputres, dim, name="resnet"):
    with tf.variable_scope(name):
        # 填充
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = general_conv2d(out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c1")
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = general_conv2d(out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c2", do_relu=False)

        return tf.nn.relu(out_res + inputres)


# 定义generator
def build_generator_resnet_9blocks(inputgen, name="generator"):
    with tf.variable_scope(name):
        f = 7
        ks = 3

        pad_input = tf.pad(inputgen, [[0, 0], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
        o_c1 = general_conv2d(pad_input, ngf, f, f, 1, 1, 0.02, name="c1")
        o_c2 = general_conv2d(o_c1, ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c2")
        o_c3 = general_conv2d(o_c2, ngf * 4, ks, ks, 2, 2, 0.02, "SAME", "c3")

        o_r1 = build_resnet_block(o_c3, ngf * 4, "r1")
        o_r2 = build_resnet_block(o_r1, ngf * 4, "r2")
        o_r3 = build_resnet_block(o_r2, ngf * 4, "r3")
        o_r4 = build_resnet_block(o_r3, ngf * 4, "r4")
        o_r5 = build_resnet_block(o_r4, ngf * 4, "r5")
        o_r6 = build_resnet_block(o_r5, ngf * 4, "r6")
        o_r7 = build_resnet_block(o_r6, ngf * 4, "r7")
        o_r8 = build_resnet_block(o_r7, ngf * 4, "r8")
        o_r9 = build_resnet_block(o_r8, ngf * 4, "r9")

        o_c4 = general_deconv2d(o_r9, [batch_size, 128, 128, ngf * 2], ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c4")
        o_c5 = general_deconv2d(o_c4, [batch_size, 256, 256, ngf], ngf, ks, ks, 2, 2, 0.02, "SAME", "c5")
        o_c6 = general_conv2d(o_c5, image_channel, f, f, 1, 1, 0.02, "SAME", "c6", do_relu=False)

        # Adding the tanh layer

        out_gen = tf.nn.tanh(o_c6, "t1")

        return out_gen


# 定义discriminator 鉴别器
def build_gen_discriminator(inputdisc, name="discriminator"):
    with tf.variable_scope(name):
        f = 4

        o_c1 = general_conv2d(inputdisc, ndf, f, f, 2, 2, 0.02, "SAME", "c1", do_norm=False, relufactor=0.2)
        o_c2 = general_conv2d(o_c1, ndf * 2, f, f, 2, 2, 0.02, "SAME", "c2", relufactor=0.2)
        o_c3 = general_conv2d(o_c2, ndf * 4, f, f, 2, 2, 0.02, "SAME", "c3", relufactor=0.2)
        o_c4 = general_conv2d(o_c3, ndf * 8, f, f, 1, 1, 0.02, "SAME", "c4", relufactor=0.2)
        o_c5 = general_conv2d(o_c4, 1, f, f, 1, 1, 0.02, "SAME", "c5", do_norm=False, do_relu=False)

        return o_c5
