"""
1-用条件数据训练CDCGAN
"""

import tensorflow as tf
import numpy as np
import random
import os
import math
import matplotlib.pyplot as plt

from A_6DataEnhance import select_axis


# 将数据与标签分离，并对标签进行one_hot编码
def input_data(finaldata):
    data = []
    label = []

    for i in range(len(finaldata)):
        data.append(finaldata[i][0])
        label.append(finaldata[i][1])
    data = np.array(data)

    data = data.reshape(-1, width, height, 1)
    label = np.array(label)
    return data, label


def CoordTrans(part_len, angle):
    Coord_x = [0]
    Coord_y = [0]
    for i in range(len(part_len)):
        rad = angle[i] * 2 * math.pi
        Coord_x.append(Coord_x[-1] + part_len[i] * math.cos(rad))
        Coord_y.append(Coord_y[-1] + part_len[i] * math.sin(rad))
    return Coord_x, Coord_y


def deconv(inputs, shape, strides, out_shape, is_sn=False, padding="SAME"):
    filters = tf.get_variable("kernel", shape=shape, initializer=tf.random_normal_initializer(stddev=0.02))
    bias = tf.get_variable("bias", shape=[shape[-2]], initializer=tf.constant_initializer([0]))
    if is_sn:
        return tf.nn.conv2d_transpose(inputs, spectral_norm("sn", filters), out_shape, strides, padding) + bias
    else:
        return tf.nn.conv2d_transpose(inputs, filters, out_shape, strides, padding) + bias


def conv(inputs, shape, strides, is_sn=False, padding="SAME"):
    filters = tf.get_variable("kernel", shape=shape, initializer=tf.random_normal_initializer(stddev=0.02))
    bias = tf.get_variable("bias", shape=[shape[-1]], initializer=tf.constant_initializer([0]))
    if is_sn:
        return tf.nn.conv2d(inputs, spectral_norm("sn", filters), strides, padding) + bias
    else:
        return tf.nn.conv2d(inputs, filters, strides, padding) + bias


def fully_connected(inputs, num_out, is_sn=False):
    W = tf.get_variable("W", [inputs.shape[-1], num_out], initializer=tf.random_normal_initializer(stddev=0.02))
    b = tf.get_variable("b", [num_out], initializer=tf.constant_initializer([0]))
    if is_sn:
        return tf.matmul(inputs, spectral_norm("sn", W)) + b
    else:
        return tf.matmul(inputs, W) + b


def leaky_relu(inputs, slope=0.2):
    return tf.maximum(slope * inputs, inputs)


def spectral_norm(name, w, iteration=1):
    # Spectral normalization which was published on ICLR2018, please refer to
    # "https://www.researchgate.net/publication/318572189_Spectral_Normalization_for_Generative_Adversarial_Networks"
    # This function spectral_norm is forked from "https://github.com/taki0112/Spectral_Normalization-Tensorflow"
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    with tf.variable_scope(name, reuse=False):
        u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
    u_hat = u
    v_hat = None

    def l2_norm(v, eps=1e-12):
        return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)
        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma
    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)
    return w_norm


def bn(inputs):  # Instance normalization？
    mean, var = tf.nn.moments(inputs, axes=[1, 2], keep_dims=True)
    scale = tf.get_variable("scale", shape=mean.shape, initializer=tf.constant_initializer([1.0]))
    shift = tf.get_variable("shift", shape=mean.shape, initializer=tf.constant_initializer([0.0]))
    return (inputs - mean) * scale / (tf.sqrt(var + epsilon)) + shift


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


class Generator:
    def __init__(self, name):
        self.name = name

    def __call__(self, Z, label, reuse=False):
        with tf.variable_scope(name_or_scope=self.name, reuse=reuse):
            # linear
            print(label.shape)
            Z = tf.concat([Z, label], 1)
            print("g_inputs:", Z.shape)
            label_ = tf.reshape(label, [batch_size, 1, 1, classfied_num])

            with tf.variable_scope(name_or_scope="train"):
                # fully connected layer for generator
                with tf.variable_scope(name_or_scope="gfc"):
                    output = fully_connected(Z, 5 * 4 * 1024)
                    output = tf.nn.relu(output)
                    output = tf.reshape(output, [batch_size, 5, 4, 1024])
                    output = conv_cond_concat(output, label_)
                    print("g_fc:", output)

                # deconv1
                # deconv(inputs, filter_shape, strides, out_shape, is_sn, padding="SAME")
                with tf.variable_scope(name_or_scope="deconv1"):
                    output = deconv(output, [3, 3, 512, (1024 + classfied_num)], [1, 1, 1, 1], [batch_size, 5, 4, 512],
                                    padding="SAME")
                    output = bn(output)
                    output = tf.nn.relu(output)
                    #                 output = conv_cond_concat(output,label_)
                    print("g_deconv1:", output)

            # deconv2
            with tf.variable_scope(name_or_scope="deconv2"):
                output = deconv(output, [3, 3, 256, 512], [1, 1, 2, 1], [batch_size, 5, 8, 256], padding="SAME")
                output = bn(output)
                output = tf.nn.relu(output)
                #                 output = conv_cond_concat(output,label_)
                print("g_deconv2:", output)

            # deconv3
            with tf.variable_scope(name_or_scope="deconv3"):
                output = deconv(output, [3, 3, 128, 256], [1, 1, 2, 1], [batch_size, 5, 16, 128], padding="SAME")
                output = bn(output)
                output = tf.nn.relu(output)
                #                 output = conv_cond_concat(output,label_)
                print("g_deconv3:", output)

            # deconv4
            with tf.variable_scope(name_or_scope="deconv4"):
                output = deconv(output, [3, 3, channel, 128], [1, 2, 2, 1], [batch_size, width, height, channel],
                                padding="SAME")
                output = tf.nn.tanh(output)
                print("g_deconv4:", output)

            return output

    @property
    def var(self):
        # 生成器所有变量
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)


class Discriminator:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, label, reuse=False, is_sn=False):
        with tf.variable_scope(name_or_scope=self.name, reuse=reuse):

            print("d_inputs:", inputs.shape)
            label = tf.reshape(label, [batch_size, 1, 1, classfied_num])
            print(label.shape)
            inputs = conv_cond_concat(inputs, label)
            print("after_concat:", inputs.shape)

            # conv1
            # conv(inputs, filter_shape, strides, is_sn, padding="SAME")
            with tf.variable_scope("conv1"):
                output = conv(inputs, [3, 3, (1 + classfied_num), 128], [1, 2, 2, 1], is_sn, padding="SAME")
                if GAN_type != "WGAN-GP":
                    output = bn(output)  # 生成器输出层,判别器输入层不用bn
                output = leaky_relu(output)
                print("d_conv1:", output)

            # conv2
            with tf.variable_scope("conv2"):
                output = conv(output, [3, 3, 128, 256], [1, 1, 2, 1], is_sn, padding="SAME")
                if GAN_type != "WGAN-GP":
                    output = bn(output)
                output = leaky_relu(output)
                print("d_conv2:", output)

            # conv3
            with tf.variable_scope("conv3"):
                output = conv(output, [3, 3, 256, 512], [1, 1, 2, 1], is_sn, padding="SAME")
                if GAN_type != "WGAN-GP":
                    output = bn(output)
                output = leaky_relu(output)
                print("d_conv3:", output)

            with tf.variable_scope(name_or_scope="train"):
                # conv4
                with tf.variable_scope("conv4"):
                    output = conv(output, [3, 3, 512, 1024], [1, 1, 1, 1], is_sn, padding="SAME")
                    output = bn(output)
                    output = leaky_relu(output)
                    print("d_conv4:", output)

                with tf.variable_scope(name_or_scope="dfc"):
                    output = tf.contrib.layers.flatten(output)
                    output = fully_connected(output, 1, is_sn)
                    print("d_fc:", output)

            return output

    @property
    def var(self):
        # 判别器所有变量
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)


class GAN:
    # Architecture of generator and discriminator just like DCGAN.
    def __init__(self):
        self.Z = tf.placeholder("float", [batch_size, 100])
        self.img = tf.placeholder("float", [batch_size, width, height, channel])
        self.label = tf.placeholder("float", [batch_size, classfied_num])
        D = Discriminator("discriminator")
        G = Generator("generator")
        self.fake_img = G(self.Z, self.label)

        # SNGAN, paper: SPECTRAL NORMALIZATION FOR GENERATIVE ADVERSARIAL NETWORKS
        self.real_logit = D(self.img, self.label, is_sn=True)
        self.fake_logit = D(self.fake_img, self.label, reuse=True, is_sn=True)

        # D_loss
        self.real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_logit, labels=tf.ones_like(self.real_logit)))
        self.fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logit, labels=tf.zeros_like(self.fake_logit)))
        self.d_loss = tf.add(self.fake_loss, self.real_loss)

        # G_loss
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logit, labels=tf.ones_like(self.fake_logit)))

        # Optimizer
        # 沈：之前这个beta1没动过，这里修改一下试试。
        self.opt_D = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.d_loss, var_list=D.var)
        self.opt_G = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.g_loss, var_list=G.var)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep = 1000)  # 沈：静脉的训练效果不好，多保存几轮作为观察

    def __call__(self):

        for e in range(epochs):

            for i in range(len(data_train) // batch_size - 1):
                batch = data_train[i * batch_size:(i + 1) * batch_size, :, :, :]
                label_batch = label_train[i * batch_size:(i + 1) * batch_size, :]
                batch = batch * 2 - 1

                z = np.random.uniform(-1, 1, (batch_size, 100)).astype(np.float32)

                # for loop in range(1):
                g_loss, _ = self.sess.run([self.g_loss, self.opt_G],
                                          feed_dict={self.img: batch, self.Z: z, self.label: label_batch})
                d_loss, _ = self.sess.run([self.d_loss, self.opt_D],
                                          feed_dict={self.img: batch, self.Z: z, self.label: label_batch})

                if i % 10 == 0:
                    print("epoch: %d, step: [%d / %d], d_loss: %g, g_loss: %g" % (
                        e, i, len(data_train) // batch_size, d_loss, g_loss))

                losses.append((d_loss, g_loss))

            # 保存每一轮的图片
            z = np.random.uniform(-1, 1, (batch_size, 100)).astype(np.float32)
            label_batch2 = np.zeros((batch_size, classfied_num))

            for i in range(len(label_batch2)):
                label_index_1 = random.randint(0, len(diam_list) - 1)
                label_index_2 = random.randint(0, len(len_list) - 1)
                label_batch2[i][0] = diam_list[label_index_1]
                label_batch2[i][1] = len_list[label_index_2]

            imgs = self.sess.run(self.fake_img, feed_dict={self.Z: z, self.label: label_batch2})
            imgs = (imgs + 1) / 2
            imgs = imgs.reshape(-1, width, height)
            print(imgs.shape)
            imgs = select_axis(imgs)
            self.save_epoch(imgs, e)

            if e >= 50:
                self.saver.save(self.sess, "../Model_Checkpoint/%s_Checkpoint/Model_%d.ckpt" % (model_name,e))

            if e % 20 == 0:
                self.plot_loss(losses)
                print("保存一次误差曲线")
        self.plot_loss(losses)
        self.sess.close()

    # 可视化单个分叉4*4
    def save_epoch(self, gen_data, e):
        gen_data = gen_data[0:16]
        r, c = 4, 4
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                img = gen_data[cnt]
                zhu = img[0]
                zuo = img[1]
                you = img[2]
                min_tag = img[3]

                zhu_diam = zhu[-1]
                bif_expo = zuo[-1]
                asy_ratio = you[-1]
                zhu_len = zhu[0:len(zhu) // 2]
                zuo_len = zuo[0:len(zuo) // 2]
                you_len = you[0:len(you) // 2]
                zhu_angle = zhu[len(zhu) // 2:(len(zhu) - 1)]
                zuo_angle = zuo[len(zuo) // 2:(len(zuo) - 1)]
                you_angle = you[len(you) // 2:(len(you) - 1)]
                tag = min_tag[0:2].index(max(min_tag[0:2]))

                # 坐标转换
                zhu_x, zhu_y = CoordTrans(zhu_len, zhu_angle)
                zuo_x, zuo_y = CoordTrans(zuo_len, zuo_angle)
                you_x, you_y = CoordTrans(you_len, you_angle)

                # 原始分叉指数获得及子血管段管径计算
                BE = bif_expo * 4.5 + 1.5
                major_diam = zhu_diam / (asy_ratio ** BE + 1) ** (1.0 / BE)
                minor_diam = major_diam * asy_ratio

                if tag == 0:
                    zuo_diam = minor_diam
                    you_diam = major_diam
                elif tag == 1:
                    zuo_diam = major_diam
                    you_diam = minor_diam

                # 画图
                axs[i, j].plot(zhu_x, zhu_y, color='red', linewidth=10 * zhu_diam)
                axs[i, j].plot(zuo_x, zuo_y, color='green', linewidth=10 * zuo_diam)
                axs[i, j].plot(you_x, you_y, color='blue', linewidth=10 * you_diam)
                axs[i, j].axis('off')
                cnt += 1
        if not os.path.exists("../Model_Checkpoint/%s_EpochImg" % model_name):
            os.makedirs("../Model_Checkpoint/%s_EpochImg"% model_name)
        plt.savefig("../Model_Checkpoint/%s_EpochImg/Epoch%d.jpg" % (model_name,e))
        plt.close()

    # 画loss曲线
    def plot_loss(self, loss):
        _, _ = plt.subplots(figsize=(20, 7))
        losses = np.array(loss)
        plt.plot(losses.T[0], label="Discriminator Loss")
        plt.plot(losses.T[1], label="Generator Loss")
        plt.title("Training Losses")
        plt.legend()
        plt.savefig('../Model_Checkpoint/%s_Loss.jpg' % model_name)
        # plt.show()

    def model_apply(self, modelpath):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(modelpath))  # 沈：加载模型数据画图

        diam_index = 0
        len_index = 0
        concat_gen = []
        for e in range(len(diam_list)*len(len_list)):
            if diam_index >= len(diam_list):
                diam_index = 0
                len_index += 1

            for loop in range(4):
                z = np.random.uniform(-1, 1, (batch_size, 100)).astype(np.float32)
                label_batch = np.zeros((batch_size, classfied_num))

                for i in range(int(batch_size)):
                    label_batch[i][0] = diam_list[diam_index]
                    label_batch[i][1] = len_list[len_index]
                gen_img = self.sess.run(self.fake_img, feed_dict={self.Z: z, self.label: label_batch})
                gen_img = gen_img.reshape(-1, width, height)
                gen_img = (gen_img + 1) / 2

                for j in range(len(gen_img)):
                    concat_gen.append(gen_img[j])
            diam_index += 1

        gen_img = np.array(concat_gen)

        self.sess.close()
        return gen_img

    def model_plot(self,modelpath):
        self.saver.restore(self.sess, modelpath)

        diam_index = 0
        len_index = 0

        fig, axs = plt.subplots(len(diam_list), len(len_list),figsize=(len(diam_list), len(len_list)))

        for e in range(len(diam_list)*len(len_list)):
            # 确定标签
            if diam_index >= len(diam_list):
                diam_index = 0
                len_index += 1

            z = np.random.uniform(-1, 1, (batch_size, 100)).astype(np.float32)
            label_batch = np.zeros((batch_size, classfied_num))

            for i in range(int(batch_size)):
                label_batch[i][0] = diam_list[diam_index]
                label_batch[i][1] = len_list[len_index]

            # 生成图片
            gen_img = self.sess.run(self.fake_img, feed_dict={self.Z: z, self.label: label_batch})
            gen_img = gen_img.reshape(-1, width, height)
            gen_img = (gen_img+1)/2
            gen_img = select_axis(gen_img)  # fencha（list4） = [zhu]=l+A+D + [zuo]=l+A+e + [you]=l+A+r + [min_tag]

            # 画图
            img = gen_img[0]
            zhu = img[0]
            zuo = img[1]
            you = img[2]
            min_tag = img[3]

            zhu_diam = zhu[-1]
            bif_expo = zuo[-1]
            asy_ratio = you[-1]
            zhu_len = zhu[0:len(zhu) // 2]
            zuo_len = zuo[0:len(zuo) // 2]
            you_len = you[0:len(you) // 2]
            zhu_angle = zhu[len(zhu) // 2:(len(zhu) - 1)]
            zuo_angle = zuo[len(zuo) // 2:(len(zuo) - 1)]
            you_angle = you[len(you) // 2:(len(you) - 1)]
            tag = min_tag[0:2].index(max(min_tag[0:2]))

            # 坐标转换
            zhu_x, zhu_y = CoordTrans(zhu_len, zhu_angle)
            zuo_x, zuo_y = CoordTrans(zuo_len, zuo_angle)
            you_x, you_y = CoordTrans(you_len, you_angle)

            # 原始分叉指数获得及子血管段管径计算
            BE = bif_expo * 4.5 + 1.5
            major_diam = zhu_diam / (asy_ratio ** BE + 1) ** (1.0 / BE)
            minor_diam = major_diam * asy_ratio

            if tag == 0:
                zuo_diam = minor_diam
                you_diam = major_diam
            elif tag == 1:
                zuo_diam = major_diam
                you_diam = minor_diam

            # 画图
            axs[diam_index, len_index].plot(zhu_x, zhu_y, color='red', linewidth=10 * zhu_diam)
            axs[diam_index, len_index].plot(zuo_x, zuo_y, color='green', linewidth=10 * zuo_diam)
            axs[diam_index, len_index].plot(you_x, you_y, color='blue', linewidth=10 * you_diam)
            axs[diam_index, len_index].axis('off')

            diam_index += 1
        plt.savefig("../ModelPlot.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"  # 选择使用的GPU

    # 参数设置
    # net, model_name = 'Convergence', 'PreTrain'
    net, model_name = 'Bifurcation', 'ArtTrain'
    classfied_num = 2  # 条件个数
    width = 10
    height = 32
    channel = 1
    GAN_type = "DSNGAN"
    batch_size = 128
    epochs = 600
    epsilon = 1e-14  # if epsilon is too big, training of DCGAN is failure.
    losses = []

    diam_list = [-1, -0.5, 0, 0.5, 1]
    len_list = [-1, 0, 1]

    # 读取数据
    pairdata_condition = np.load('../Data/Normalized Data/%s_Data&Condition.npy' % net, allow_pickle=True)
    print(pairdata_condition.shape)

    # 打乱数据
    np.random.shuffle(pairdata_condition)

    # 训练集
    data_train, label_train = input_data(pairdata_condition)
    print(data_train.shape)
    print(label_train.shape)

    # 模型训练
    tf.reset_default_graph()
    PreTrain = GAN()
    PreTrain()



