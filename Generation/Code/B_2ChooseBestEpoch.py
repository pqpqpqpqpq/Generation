"""
挑选合适的epoch，输出显著性检验P>0.01的轮次
"""

import tensorflow as tf
import numpy as np
import random
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from C_Func import tort
from A_6DataEnhance import select_axis


def cross(a, b, c):  # 跨立实验
    x1 = b[0] - a[0]
    y1 = b[1] - a[1]
    x2 = c[0] - a[0]
    y2 = c[1] - a[1]
    return x1 * y2 - x2 * y1


def IsIntersec(a1, a2, b1, b2):  # 判断两线段是否相交

    if (max(a1[0], a2[0]) >= min(b1[0], b2[0])  # 矩形1最右端大于矩形2最左端
            and max(b1[0], b2[0]) >= min(a1[0], a2[0])  # 矩形2最右端大于矩形最左端
            and max(a1[1], a2[1]) >= min(b1[1], b2[1])  # 矩形1最高端大于矩形最低端
            and max(b1[1], b2[1]) >= min(a1[1], a2[1])):  # 矩形2最高端大于矩形最低端
        if cross(a1, a2, b1) * cross(a1, a2, b2) <= 0 and cross(b1, b2, a1) * cross(b1, b2, a2) <= 0:
            return True
        else:
            return False
    else:
        return False


# 碰撞自检函数(self_data: N*10*32, gen_well_data: M*10*32)
def CrashBranchRemove(data):
    gen_well_data = []
    for loop in range(len(data)):
        img = data[loop]
        zhu = img[0]
        zuo = img[1]
        you = img[2]

        zhu_len = zhu[0:len(zhu) // 2]
        zuo_len = zuo[0:len(zuo) // 2]
        you_len = you[0:len(you) // 2]
        zhu_angle = zhu[len(zhu) // 2:(len(zhu) - 1)]
        zuo_angle = zuo[len(zuo) // 2:(len(zuo) - 1)]
        you_angle = you[len(you) // 2:(len(you) - 1)]

        # 坐标转换
        zhu_x, zhu_y = CoordTrans(zhu_len, zhu_angle)
        zuo_x, zuo_y = CoordTrans(zuo_len, zuo_angle)
        you_x, you_y = CoordTrans(you_len, you_angle)

        # 所有节点信息
        zhu_list = []
        for i in range(len(zhu_x) - 1):
            zhu_list.append([[zhu_x[i], zhu_y[i]], [zhu_x[i + 1], zhu_y[i + 1]]])

        zuo_list = []
        for i in range(len(zuo_x) - 1):
            zuo_list.append([[zuo_x[i], zuo_y[i]], [zuo_x[i + 1], zuo_y[i + 1]]])

        you_list = []
        for i in range(len(you_x) - 1):
            you_list.append([[you_x[i], you_y[i]], [you_x[i + 1], you_y[i + 1]]])

        Coord_List1 = zhu_list + zuo_list + you_list
        Coord_List2 = zhu_list + zuo_list + you_list

        # 判断血管段分段（线段）是否相交
        BoolenList = []
        for i in range(len(Coord_List1)):
            for j in range(len(Coord_List2)):
                # 首先剔除端点相同的血管段
                if Coord_List2[j][0] == Coord_List1[i][0] or Coord_List2[j][0] == Coord_List1[i][1] or Coord_List2[j][
                    1] == \
                        Coord_List1[i][0] or Coord_List2[j][1] == Coord_List1[i][1]:
                    continue

                # 判断是否相交
                Boolen = IsIntersec(Coord_List1[j][0], Coord_List1[j][1], Coord_List2[i][0], Coord_List2[i][1])
                BoolenList.append(Boolen)

        if True not in BoolenList:
            gen_well_data.append(data[loop])
    return gen_well_data

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


# 坐标转换
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


def bn(inputs):
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
            # print(label.shape)
            Z = tf.concat([Z, label], 1)
            # print("g_inputs:", Z.shape)
            label_ = tf.reshape(label, [batch_size, 1, 1, classfied_num])

            with tf.variable_scope(name_or_scope="train"):
                # fully connected layer for generator
                with tf.variable_scope(name_or_scope="gfc"):
                    output = fully_connected(Z, 5 * 4 * 1024)
                    output = tf.nn.relu(output)
                    output = tf.reshape(output, [batch_size, 5, 4, 1024])
                    output = conv_cond_concat(output, label_)
                    # print("g_fc:", output)

                # deconv1
                # deconv(inputs, filter_shape, strides, out_shape, is_sn, padding="SAME")
                with tf.variable_scope(name_or_scope="deconv1"):
                    output = deconv(output, [3, 3, 512, (1024 + classfied_num)], [1, 1, 1, 1], [batch_size, 5, 4, 512],
                                    padding="SAME")
                    output = bn(output)
                    output = tf.nn.relu(output)
                    #
                    # print("g_deconv1:", output)

            # deconv2
            with tf.variable_scope(name_or_scope="deconv2"):
                output = deconv(output, [3, 3, 256, 512], [1, 1, 2, 1], [batch_size, 5, 8, 256], padding="SAME")
                output = bn(output)
                output = tf.nn.relu(output)
                #
                # print("g_deconv2:", output)

            # deconv3
            with tf.variable_scope(name_or_scope="deconv3"):
                output = deconv(output, [3, 3, 128, 256], [1, 1, 2, 1], [batch_size, 5, 16, 128], padding="SAME")
                output = bn(output)
                output = tf.nn.relu(output)
                #
                # print("g_deconv3:", output)

            # deconv4
            with tf.variable_scope(name_or_scope="deconv4"):
                output = deconv(output, [3, 3, channel, 128], [1, 2, 2, 1], [batch_size, width, height, channel],
                                padding="SAME")
                output = tf.nn.tanh(output)
                # print("g_deconv4:", output)

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

            # print("d_inputs:", inputs.shape)
            label = tf.reshape(label, [batch_size, 1, 1, classfied_num])
            # print(label.shape)
            inputs = conv_cond_concat(inputs, label)
            # print("after_concat:", inputs.shape)

            # conv1
            # conv(inputs, filter_shape, strides, is_sn, padding="SAME")
            with tf.variable_scope("conv1"):
                output = conv(inputs, [3, 3, (1 + classfied_num), 128], [1, 2, 2, 1], is_sn, padding="SAME")
                if GAN_type != "WGAN-GP":
                    output = bn(output)  # 生成器输出层,判别器输入层不用bn
                output = leaky_relu(output)
                # print("d_conv1:", output)

            # conv2
            with tf.variable_scope("conv2"):
                output = conv(output, [3, 3, 128, 256], [1, 1, 2, 1], is_sn, padding="SAME")
                if GAN_type != "WGAN-GP":
                    output = bn(output)
                output = leaky_relu(output)
                # print("d_conv2:", output)

            # conv3
            with tf.variable_scope("conv3"):
                output = conv(output, [3, 3, 256, 512], [1, 1, 2, 1], is_sn, padding="SAME")
                if GAN_type != "WGAN-GP":
                    output = bn(output)
                output = leaky_relu(output)
                # print("d_conv3:", output)

            with tf.variable_scope(name_or_scope="train"):
                # conv4
                with tf.variable_scope("conv4"):
                    output = conv(output, [3, 3, 512, 1024], [1, 1, 1, 1], is_sn, padding="SAME")
                    output = bn(output)
                    output = leaky_relu(output)
                    # print("d_conv4:", output)

                with tf.variable_scope(name_or_scope="dfc"):
                    output = tf.contrib.layers.flatten(output)
                    output = fully_connected(output, 1, is_sn)
                    # print("d_fc:", output)

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
        self.opt_D = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.d_loss, var_list=D.var)
        self.opt_G = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.g_loss, var_list=G.var)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=50)

    def model_apply(self, modelpath):
        self.saver.restore(self.sess, modelpath)
        print('处理生成数据：')
        print('加载生成模型：', modelpath)
        diam_index = 0
        len_index = 0
        concat_gen = []
        for e in range(len(diam_list) * len(len_list)):
            if diam_index >= len(diam_list):
                diam_index = 0
                len_index += 1

            for loop in range(4):
                z = np.random.uniform(-1, 1, (batch_size, 100)).astype(np.float32)
                label_batch = np.zeros((batch_size, classfied_num))

                for i in range(int(batch_size)):
                    label_batch[i][0] = diam_list[diam_index]
                    label_batch[i][1] = len_list[len_index]
                print('加载生成数据', 'label:', label_batch[0], '生成完成')
                gen_img = self.sess.run(self.fake_img, feed_dict={self.Z: z, self.label: label_batch})
                gen_img = gen_img.reshape(-1, width, height)
                gen_img = (gen_img + 1) / 2

                for j in range(len(gen_img)):
                    concat_gen.append(gen_img[j])
            diam_index += 1
        # self.sess.close()

        gen_img = np.array(concat_gen)
        DataSet = select_axis(gen_img)
        DataSet = CrashBranchRemove(DataSet)  # 筛除碰撞的生成分叉

        gen_Diam_mat = []
        gen_Len_mat = []
        gen_Tort_mat = []
        gen_K_mat = []
        gen_lambda_mat = []
        gen_angle_mat = []

        for i in range(len(DataSet)):
            Data = DataSet[i]

            zhu = Data[0]
            zuo = Data[1]
            you = Data[2]

            zhu_diam = zhu[-1]
            bif_expo = zuo[-1]
            asy_ratio = you[-1]
            zhu_len = zhu[0:len(zhu) // 2]

            zhu_angle = zhu[len(zhu) // 2:(len(zhu) - 1)]
            zuo_angle = zuo[len(zuo) // 2:(len(zuo) - 1)]
            you_angle = you[len(you) // 2:(len(you) - 1)]

            a1 = abs(zuo_angle[0] - you_angle[0]) * 360
            if a1 < 180:
                gen_angle_mat.append(a1)
            else:
                gen_angle_mat.append(360 - a1)

            # 实际值
            gen_Diam_mat.append(zhu_diam * 70)
            gen_Len_mat.append(sum(zhu_len) * 200)
            gen_Tort_mat.append(tort(zhu_len, zhu_angle))
            gen_K_mat.append(bif_expo * 4.5 + 1.5)
            gen_lambda_mat.append(asy_ratio)

        print("生成数据特征：==============================")
        print("生成管径：", np.mean(gen_Diam_mat), np.std(gen_Diam_mat))
        print("生成管长：", np.mean(gen_Len_mat), np.std(gen_Len_mat))
        print("生成卷曲度：", np.mean(gen_Tort_mat), np.std(gen_Tort_mat))
        print("生成分叉指数：", np.mean(gen_K_mat), np.std(gen_K_mat))
        print("生成不对称比例：", np.mean(gen_lambda_mat), np.std(gen_lambda_mat))
        print("生成分叉角：", np.mean(gen_angle_mat), np.std(gen_angle_mat))

        # 返回归一化的值用于计算FID
        return (gen_Diam_mat, gen_Len_mat, gen_Tort_mat, gen_K_mat, gen_lambda_mat, gen_angle_mat,
                np.mean(gen_Diam_mat), np.std(gen_Diam_mat),
                np.mean(gen_Len_mat), np.std(gen_Len_mat),
                np.mean(gen_Tort_mat), np.std(gen_Tort_mat),
                np.mean(gen_K_mat), np.std(gen_K_mat),
                np.mean(gen_lambda_mat), np.std(gen_lambda_mat),
                np.mean(gen_angle_mat), np.std(gen_angle_mat)
                )

    def ChooseBestStatistic(self, modelpath, n):
        Diam_mat = []
        Len_mat = []
        Tort_mat = []
        K_mat = []
        lambda_mat = []
        angle_mat = []
        print('处理原始数据：')
        for net in Net:
            for nt in NetType:
                DataSet = np.load("../Data/Normalized Data/Norm_%s_%s.npy" % (net, nt))
                print('加载原始数据：',"Normalized Data/Norm_%s_%s.npy" % (net, nt))
                DataSet = select_axis(DataSet)
                for i in range(len(DataSet)):
                    Data = DataSet[i]

                    zhu = Data[0]
                    zuo = Data[1]
                    you = Data[2]

                    zhu_diam = zhu[-1]
                    bif_expo = zuo[-1]
                    asy_ratio = you[-1]
                    zhu_len = zhu[0:len(zhu) // 2]
                    zhu_angle = zhu[len(zhu) // 2:(len(zhu) - 1)]
                    zuo_angle = zuo[len(zuo) // 2:(len(zuo) - 1)]
                    you_angle = you[len(you) // 2:(len(you) - 1)]

                    a = abs(zuo_angle[0] - you_angle[0]) * 360
                    if a < 180:
                        angle_mat.append(a)
                    else:
                        angle_mat.append(360 - a)
                    # 实际值
                    Diam_mat.append(zhu_diam * 70)
                    Len_mat.append(sum(zhu_len) * 200)
                    Tort_mat.append(tort(zhu_len, zhu_angle))  # 卷曲度
                    K_mat.append(bif_expo * 4.5 + 1.5)
                    lambda_mat.append(asy_ratio)

        print("原始数据特征：==============================")
        print("管径：", np.mean(Diam_mat), np.std(Diam_mat))
        print("管长：", np.mean(Len_mat), np.std(Len_mat))
        print("卷曲度：", np.mean(Tort_mat), np.std(Tort_mat))
        print("分叉指数：", np.mean(K_mat), np.std(K_mat))
        print("不对称比例：", np.mean(lambda_mat), np.std(lambda_mat))
        print("角度：", np.mean(angle_mat), np.std(angle_mat))

        # 模型训练
        tf.reset_default_graph()
        TestModel = GAN()

        (gen_Diam_mat, gen_Len_mat, gen_Tort_mat, gen_K_mat, gen_lambda_mat, gen_angel_mat,
         meanDiam, stdDiam,
         meanLen_mat, std_Len,
         meanTort, stdTort,
         meanK_mat, stdK_mat,
         meanlambda, stdlambda,
         meanAngle, stdAngle
         ) = TestModel.model_apply(modelpath)


        t_diam, p_diam = stats.ttest_ind(Diam_mat, gen_Diam_mat)
        t_len, p_len = stats.ttest_ind(Len_mat, gen_Len_mat)
        t_tort, p_tort = stats.ttest_ind(Tort_mat, gen_Tort_mat)
        t_k, p_k = stats.ttest_ind(K_mat, gen_K_mat)
        t_lambda, p_lambda = stats.ttest_ind(lambda_mat, gen_lambda_mat)
        t_angle, p_angle = stats.ttest_ind(angle_mat, gen_angel_mat)

        print("是否有显著差异及p值：===========")
        print("管径：", t_diam, p_diam)
        print("管长：", t_len, p_len)
        print("卷曲度：", t_tort, p_tort)
        print("分叉指数：", t_k, p_k)
        print("不对称比例：", t_lambda, p_lambda)
        print('角度：', t_angle, p_angle)

        if p_diam > 0.01 and p_len > 0.01 and p_tort > 0.01 \
                and p_k > 0.01 and p_lambda > 0.01 and p_angle > 0.01:
            return n*2+50
        else:
            return 0


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"  # 选择使用的GPU

    # 参数设置

    classfied_num = 2  # 条件个数
    width = 10
    height = 32
    channel = 1
    GAN_type = "DSNGAN"  # DCGAN, DSNGAN,RDSGAN, RDSNGAN, LSGAN, LSNGAN, RLSGAN, RLSNGAN, WGAN, WGAN-GP
    batch_size = 128
    epsilon = 1e-14  # if epsilon is too big, training of DCGAN is failure.
    losses = []

    diam_list = [-1, -0.5, 0, 0.5, 1]
    len_list = [-1, 0, 1]

    tf.reset_default_graph()
    PreTrain = GAN()  # GAN类的__init__
    # NetType, Check = ['Convergence'], 'PreTrain_Checkpoint'
    NetType, Check = ['Bifurcation'], 'ArtTrain_Checkpoint'

    BestStatistic = []
    Best_index = []

    #  观察生成数据指标，真实数据指标显著性检验结果
    for i in range(250):
        Net = ['Men_389', 'Men_546', 'Men_913']
        modelpath = ("../Model_Checkpoint/%s/Model_%d.ckpt" % (Check, i * 2 + 50))
        print('===================================================================')
        print('调用的参数：', ("../Model_Checkpoint/%s/Model_%d.ckpt" % (Check, i * 2 + 50)))
        n = PreTrain.ChooseBestStatistic(modelpath, i)
        Best_index.append(n)

    print("=========结果显著的轮次如下：=========")
    print(Best_index)



