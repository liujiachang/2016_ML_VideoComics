import tensorflow as tf
import random

pool_size = 32 #大小

# 定义leaky_relu 激活函数
def lrelu(x, leak=0.2, name="lrelu", alt_relu_impl=False):
    with tf.variable_scope(name):
        if alt_relu_impl:
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            # lrelu = 1/2 * (1 + leak) * x + 1/2 * (1 - leak) * |x|
            return f1 * x + f2 * abs(x) # 当x大于0时 该式等于x
        else:
            return tf.maximum(x, leak * x) # max函数


# 定义instance_norm层 IN主要是风格迁移中的归一化
def instance_norm(x):
    with tf.variable_scope("instance_norm"):

        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale', [x.get_shape()[-1]],initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))#系数1 初始化
        offset = tf.get_variable('offset', [x.get_shape()[-1]], initializer=tf.constant_initializer(0.0))#系数2 初始化
        x_normalized =  tf.div(x - mean, tf.sqrt(var + epsilon)) #公式
        out = scale * x_normalized + offset

        return out


# 定义卷积层conv2d tensorflow 框架
def general_conv2d(inputconv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02, padding="VALID", name="conv2d",
                   do_norm=True, do_relu=True, relufactor=0):
    with tf.variable_scope(name):

        conv = tf.contrib.layers.conv2d(inputconv, o_d, [f_h, f_w], [s_h, s_w], padding, activation_fn=None,
                                        weights_initializer=tf.truncated_normal_initializer(stddev=stddev), # 如果生成的值大于平均值2个标准偏差的值则丢弃重新选择
                                        biases_initializer=tf.constant_initializer(0.0))
        if do_norm: # 归一化
            conv = instance_norm(conv)
        if do_relu:# 激活函数
            if (relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")
        return conv


# 定义反卷积层deconv2d
def general_deconv2d(inputconv, outshape, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02, padding="VALID",
                     name="deconv2d", do_norm=True, do_relu=True, relufactor=0):
    with tf.variable_scope(name):

        deconv = tf.contrib.layers.conv2d_transpose(inputconv, o_d, [f_h, f_w], [s_h, s_w], padding, activation_fn=None,
                                                  weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                                  biases_initializer=tf.constant_initializer(0.0))

        if do_norm:
            deconv = instance_norm(deconv)
        if do_relu:
            if (relufactor == 0):
                deconv = tf.nn.relu(deconv, "relu")
            else:
                deconv = lrelu(deconv, relufactor, "lrelu")
        return deconv


# 定义image_pool函数 随机池化
def fake_image_pool(num_fakes, fake, fake_pool):
    '''
    函数功能：将num_fakes张生成器生成的影像，保存到fake_pool中
    如果图片数量太多，随机保存
    '''

    if (num_fakes < pool_size):
        fake_pool[num_fakes] = fake
        return fake
    else:
        p = random.random()
        if p > 0.5:
            random_id = random.randint(0, pool_size - 1)
            temp = fake_pool[random_id]
            fake_pool[random_id] = fake
            return temp
        else:
            return fake


