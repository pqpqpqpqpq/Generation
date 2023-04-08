"""
进行血管树的生成代码
"""

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from A_6DataEnhance import select_axis
from C_Func import CrashBranchRemove, GrowSeqSort, get_condition, get_coord_seg, Cal_Error, \
    angle_determine, Gen_Rotate, CrashDetection, SegExtract
from C_GetEntrance import CoordTrans_R
from scipy import io


def plot_tree(Tree_Data, Coord_Data, Tree_Num, path):
    x_list = []
    y_list = []

    for i in range(len(Coord_Data)):
        Coord_x = Coord_Data[i][0]
        Coord_y = Coord_Data[i][1]
        for j in Coord_x:
            x_list.append(j)
        for j in Coord_y:
            y_list.append(j)

    dpi = 600
    umpp = 1
    fig = plt.figure(figsize=((max(x_list) - min(x_list)) / dpi, (max(y_list) - min(y_list)) / dpi))
    ax = fig.add_axes([0, 0, 1, 1])

    for j in range(len(Tree_Data)):
        # linewidth单位为points，1 point = 1/72 inch, 1 inch = dpi pixels, 1 pixel = umpp um (umpp: um per pixel)
        # 则管径为D um的血管段，画图时对应的linewidth为72 /(umpp*dpi) * D
        ax.plot(Coord_Data[j][0], Coord_Data[j][1], 'k-', linewidth=72 / (dpi * umpp) * Tree_Data[j][3])

    ax.axis('off')
    plt.xlim(min(x_list), max(x_list))
    plt.ylim(min(y_list), max(y_list))

    plt.savefig('%s/Tree_%s_%d.tiff' % (path, net, Tree_Num), dpi=dpi, pil_kwargs={"compression": "tiff_lzw"})
    plt.close()


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
    # Spectral normalization which was published on ICLR2018,please refer to "https://www.researchgate.net/publication/318572189_Spectral_Normalization_for_Generative_Adversarial_Networks"
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
            Z = tf.concat([Z, label], 1)
            label_ = tf.reshape(label, [batch_size, 1, 1, classfied_num])

            with tf.variable_scope(name_or_scope="train"):
                # fully connected layer for generator
                with tf.variable_scope(name_or_scope="gfc"):
                    output = fully_connected(Z, 5 * 4 * 1024)
                    output = tf.nn.relu(output)
                    output = tf.reshape(output, [batch_size, 5, 4, 1024])
                    output = conv_cond_concat(output, label_)

                # deconv1
                # deconv(inputs, filter_shape, strides, out_shape, is_sn, padding="SAME")
                with tf.variable_scope(name_or_scope="deconv1"):
                    output = deconv(output, [3, 3, 512, (1024 + classfied_num)], [1, 1, 1, 1], [batch_size, 5, 4, 512],
                                    padding="SAME")
                    output = bn(output)
                    output = tf.nn.relu(output)

            # deconv2
            with tf.variable_scope(name_or_scope="deconv2"):
                output = deconv(output, [3, 3, 256, 512], [1, 1, 2, 1], [batch_size, 5, 8, 256], padding="SAME")
                output = bn(output)
                output = tf.nn.relu(output)

            # deconv3
            with tf.variable_scope(name_or_scope="deconv3"):
                output = deconv(output, [3, 3, 128, 256], [1, 1, 2, 1], [batch_size, 5, 16, 128], padding="SAME")
                output = bn(output)
                output = tf.nn.relu(output)

            # deconv4
            with tf.variable_scope(name_or_scope="deconv4"):
                output = deconv(output, [3, 3, channel, 128], [1, 2, 2, 1], [batch_size, width, height, channel],
                                padding="SAME")
                output = tf.nn.tanh(output)

        return output

    @property
    def var(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '/train')


class Discriminator:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, label, reuse=False, is_sn=False):
        with tf.variable_scope(name_or_scope=self.name, reuse=reuse):
            label = tf.reshape(label, [batch_size, 1, 1, classfied_num])
            inputs = conv_cond_concat(inputs, label)

            # conv1
            # conv(inputs, filter_shape, strides, is_sn, padding="SAME")
            with tf.variable_scope("conv1"):
                output = conv(inputs, [3, 3, (1 + classfied_num), 128], [1, 2, 2, 1], is_sn, padding="SAME")
                output = bn(output)  # 生成器输出层,判别器输入层不用bn
                output = leaky_relu(output)

            # conv2
            with tf.variable_scope("conv2"):
                output = conv(output, [3, 3, 128, 256], [1, 1, 2, 1], is_sn, padding="SAME")
                output = bn(output)
                output = leaky_relu(output)

            # conv3
            with tf.variable_scope("conv3"):
                output = conv(output, [3, 3, 256, 512], [1, 1, 2, 1], is_sn, padding="SAME")
                output = bn(output)
                output = leaky_relu(output)

            with tf.variable_scope(name_or_scope="train"):
                # conv4
                with tf.variable_scope("conv4"):
                    output = conv(output, [3, 3, 512, 1024], [1, 1, 1, 1], is_sn, padding="SAME")
                    output = bn(output)
                    output = leaky_relu(output)

                # fully connected layer for generator
                with tf.variable_scope(name_or_scope="dfc"):
                    output = tf.contrib.layers.flatten(output)
                    output = fully_connected(output, 1, is_sn)

            return output

    @property
    def var(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '/train')


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
        self.saver = tf.train.Saver()

    def generate_branch(self, label_mom):

        concat_gen = []
        for loop in range(10):
            z = np.random.uniform(-1, 1, (batch_size, 100)).astype(np.float32)
            label_batch = np.zeros((batch_size, classfied_num))
            for i in range(len(label_batch)):
                label_batch[i][0] = label_mom[0]
                label_batch[i][1] = label_mom[1]
            gen = self.sess.run(self.fake_img, feed_dict={self.Z: z, self.label: label_batch})
            gen = gen.reshape(-1, width, height)
            gen = (gen + 1) / 2

            for j in range(len(gen)):
                concat_gen.append(gen[j])
        gen = np.array(concat_gen)  # 得到生成的子分叉数组
        gen = select_axis(gen)
        gen = CrashBranchRemove(gen)  # 筛除碰撞的生成分叉
        return gen

    def generate_tree(self, Tree_Num, modelpath, vestype):
        # 定义参数，路径
        self.saver.restore(self.sess, modelpath)

        threshold = 6  # 截止管径（um）

        if not os.path.exists("../Data/Generated Trees/TreeImgs_N%s_T%d" % (net, Tree_Num)):
            os.makedirs("../Data/Generated Trees/TreeImgs_N%s_T%d" % (net, Tree_Num))

        # 获取入口分叉，分别生长左右分叉
        if vestype == 'ven':
            root = np.load("../Data/Normalized Data/Entrance_Men_%s_Convergence.npy" % net)  # 静脉树入口
        elif vestype == 'art':
            root = np.load("../Data/Normalized Data/Entrance_Men_%s_Bifurcation.npy" % net)  # 动脉树入口
        # angle_distribution = np.load("../Data/Normalized Data/Jiajiao.npy")
        #
        Tree_Data = []
        Coord_Data = []
        Coord_1, Coord_2, Coord_3, Seg_1, Seg_2, Seg_3 = get_coord_seg(root)  # 获取坐标

        # Coord_Data: [[[Coord_x],[Coord_y]]...]
        Coord_Data.append(Coord_1)
        Coord_Data.append(Coord_2)
        Coord_Data.append(Coord_3)

        # Tree_Data: [[ParentNode, DaughterNode1, DaughterNode2, Diam, Length, Tortuosity, Level]...]
        Tree_Data.append([0, 2, 3] + Seg_1 + [1])
        Tree_Data.append([1, 0, 0] + Seg_2 + [2])
        Tree_Data.append([1, 0, 0] + Seg_3 + [2])
        level = 2

        # 以最大level血管段管径均小于等于threshold作为循环结束的条件
        Diam_List = []
        for i in range(len(Tree_Data)):
            if Tree_Data[i][-1] == level and Tree_Data[i][3] > threshold:  # 沈：取终端的管径信息
                Diam_List.append(Tree_Data[i][3])

        while len(Diam_List) > 0:
            level += 1

            # Seg_Mom_List, Coord_Mom_List分别对应level最大的Tree_Data和Coord_Data, Index_List是将分叉按管径从大到小排列的血管段编号
            Seg_Mom_List, Coord_Mom_List, Index_List = GrowSeqSort(Tree_Data, Coord_Data)

            for i in range(len(Index_List)):
                Seg_Mom = Seg_Mom_List[Index_List[i]]
                Seg_Mom_Index = Tree_Data.index(Seg_Mom)
                Coord_Mom = Coord_Mom_List[Index_List[i]]

                if Seg_Mom[3] > threshold:
                    folder = "../Data/Generated Trees/TreeImgs_N%s_T%d/Epoch%d%d_" % (net, Tree_Num, level - 1, i)
                    condition = get_condition(Seg_Mom[3:5])  # 获取父系的label
                    centroid_dir_rev, end_angle = angle_determine(folder, Tree_Data, Coord_Data, Coord_Mom)

                    IsGrow = False
                    loop_num = 0

                    while not IsGrow and loop_num < 4:
                        loop_num += 1
                        Data_Son = gan.generate_branch(condition)
                        error_th = 0.1
                        error_mat = Cal_Error(Seg_Mom[3:6], Data_Son,
                                              error_th)

                        while len(error_mat) == 0 and error_th < 0.5:
                            error_th += 0.05
                            error_mat = Cal_Error(Seg_Mom[3:6], Data_Son, error_th)

                        for emi in range(len(error_mat)):
                            branch = Data_Son[error_mat[emi][1]]
                            zuo_len, you_len, zuo_angle, you_angle = Gen_Rotate(branch, centroid_dir_rev, end_angle)

                            start_x, start_y = Coord_Mom[0][-1], Coord_Mom[1][-1]
                            zuo_coord_x, zuo_coord_y = CoordTrans_R(start_x, start_y, np.array(zuo_len) * 200,
                                                                    zuo_angle)
                            you_coord_x, you_coord_y = CoordTrans_R(start_x, start_y, np.array(you_len) * 200,
                                                                    you_angle)

                            flag1 = CrashDetection(Coord_Data, [zuo_coord_x, zuo_coord_y])
                            flag2 = CrashDetection(Coord_Data, [you_coord_x, you_coord_y])

                            if flag1 & flag2:
                                Coord_Data.append([zuo_coord_x, zuo_coord_y])
                                Coord_Data.append([you_coord_x, you_coord_y])
                                Tree_Data[Seg_Mom_Index][1:3] = [len(Tree_Data) + 1, len(Tree_Data) + 2]
                                Seg_zuo, Seg_you = SegExtract(branch, Seg_Mom[3])
                                Tree_Data.append([Seg_Mom_Index + 1, 0, 0] + Seg_zuo + [level])
                                Tree_Data.append([Seg_Mom_Index + 1, 0, 0] + Seg_you + [level])
                                IsGrow = True
                                break
            print('已完成第', level, '层血管段的生成')
            Diam_List = []
            for i in range(len(Tree_Data)):
                if Tree_Data[i][-1] == level and Tree_Data[i][3] > threshold:
                    Diam_List.append(Tree_Data[i][3])
                    break
        print("=====已完成第", Tree_Num, "个血管树的生成=====")
        # self.sess.close()
        return Tree_Data, Coord_Data


if __name__ == "__main__":

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

    # 参数设置
    classfied_num = 2
    width = 10
    height = 32
    channel = 1
    n_class = 131
    GAN_type = "DSNGAN"
    epochs = 200
    epsilon = 1e-14
    losses = []
    batch_size=128

    diam_list = [-1, -0.5, 0, 0.5, 1]
    len_list = [-1, 0, 1]

    tf.reset_default_graph()
    gan = GAN()
    vestype = 'ven'
    # vestype = 'art'

    # 选取生成模型
    if vestype == 'ven':
        modelpath = "../Model_Checkpoint/PreTrain_Checkpoint/Model_420.ckpt"  # 静脉树模型
        path = "../Data/ven Trees"
    else:
        modelpath = "../Model_Checkpoint/ArtTrain_Checkpoint/Model_328.ckpt"  # 动脉树模型
        path = "../Data/art Trees"

    if not os.path.exists('%s' % path):
        os.makedirs('%s' % path)

    # 生成树
    net_list = ['389', '546', '913']
    tree_num = 3  # 每个入口生成 tree_num 棵树
    for net in net_list:
        for Tree_Num in range(tree_num):
            Tree_Data, Coord_Data = gan.generate_tree(Tree_Num, modelpath, vestype)

            np.save('%s/Tree_%s_%s.npy' % (path, net, Tree_Num), Tree_Data)
            io.savemat('%s/Tree_%s_%s.mat' % (path, net, Tree_Num), {'data': Tree_Data})

            np.save('%s/Tree_%s_%s_Coord.npy' % (path, net, Tree_Num), Coord_Data)
            io.savemat('%s/Tree_%s_%s_Coord.mat' % (path, net, Tree_Num), {'data': Coord_Data})

            plot_tree(Tree_Data, Coord_Data, Tree_Num, path)
