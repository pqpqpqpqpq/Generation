"""
获取原始数据的入口
"""

import numpy as np
import math
import h5py
import matplotlib.pyplot as plt
from A_5DataNormalize import angle_len_norm, diam_norm


def CoordTrans_R(x, y, part_len, angle):
    Coord_x = [x]
    Coord_y = [y]
    for i in range(len(part_len)):
        rad = angle[i] * 2 * math.pi
        Coord_x.append(Coord_x[-1] + part_len[i] * math.cos(rad))
        Coord_y.append(Coord_y[-1] + part_len[i] * math.sin(rad))
    return Coord_x, Coord_y


def angle_correct(anglelist):
    for i in range(len(anglelist)):
        if anglelist[i] < 0:
            anglelist[i] = anglelist[i] + 1
        elif anglelist[i] >= 1:
            anglelist[i] = anglelist[i] - 1
    return anglelist


# 从4行的紧凑型数据变成用于训练的10*32标准数据，可以看作是select_axis函数的反过程
def trans4210(branch):
    zhu = branch[0]
    zuo = branch[1]
    you = branch[2]
    min_tag = branch[3]

    zhu_diam = zhu[-1]
    bif_expo = zuo[-1]
    asy_ratio = you[-1]

    zhu_len = zhu[0:len(zhu) // 2]
    zuo_len = zuo[0:len(zuo) // 2]
    you_len = you[0:len(you) // 2]
    zhu_angle = zhu[len(zhu) // 2:(len(zhu) - 1)]
    zuo_angle = zuo[len(zuo) // 2:(len(zuo) - 1)]
    you_angle = you[len(you) // 2:(len(you) - 1)]

    insert_num = 32
    insert_Main_len = zhu_len + [0.] * (insert_num - len(zhu_len))  # 在加号前加#号，把前面的插值加上,即为插值
    insert_Left_len = zuo_len + [0.] * (insert_num - len(zuo_len))
    insert_Right_len = you_len + [0.] * (insert_num - len(you_len))
    insert_Main_angle = zhu_angle + [0.] * (insert_num - len(zhu_angle))
    insert_Left_angle = zuo_angle + [0.] * (insert_num - len(zuo_angle))
    insert_Right_angle = you_angle + [0.] * (insert_num - len(you_angle))
    insert_Main_diam = [zhu_diam] + [0.] * (insert_num - 1)
    insert_bif_expo = [bif_expo] + [0.] * (insert_num - 1)
    insert_asy_ratio = [asy_ratio] + [0.] * (insert_num - 1)

    insert_Main_diam[len(zhu_len)] = 1
    insert_bif_expo[len(zuo_len)] = 1
    insert_asy_ratio[len(you_len)] = 1
    num_label = min_tag

    branch_mod = [insert_Main_len] + \
             [insert_Left_len] + \
             [insert_Right_len] + \
             [insert_Main_angle] + \
             [insert_Left_angle] + \
             [insert_Right_angle] + \
             [insert_Main_diam] + \
             [insert_bif_expo] + \
             [insert_asy_ratio] + \
             [num_label]

    return np.array(branch_mod)


# 函数的目的是让分叉进行适当的旋转使得主分叉起点位于（0，0），朝向为正90度
def entrance_mod(branch):
    zhu = branch[0]
    zuo = branch[1]
    you = branch[2]
    min_tag = branch[3]

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

    # 之前的分叉的三个分支都是以交叉点为起点排布长度和角度，这里把主分叉的数据调换一下
    zhu_len_mod = zhu_len[::-1]
    tmp = zhu_angle[::-1]
    zhu_angle_mod = []
    for i in range(len(tmp)):
        if tmp[i]>=0.5:
            zhu_angle_mod.append(tmp[i]-0.5)
        else:
            zhu_angle_mod.append(tmp[i]+0.5)

    # 确定需要旋转的角度
    zhu_x, zhu_y = CoordTrans_R(0, 0, zhu_len_mod, zhu_angle_mod)
    angle_bias = 0.25 - math.atan2(zhu_y[-1] - zhu_y[0], zhu_x[-1] - zhu_x[0])/ (2*math.pi)

    zhu_angle_mod = angle_correct(np.array(zhu_angle_mod) + angle_bias)
    zuo_angle_mod = angle_correct(np.array(zuo_angle) + angle_bias)
    you_angle_mod = angle_correct(np.array(you_angle) + angle_bias)

    zhu_x, zhu_y = CoordTrans_R(0,0,zhu_len_mod,zhu_angle_mod)
    zuo_x, zuo_y = CoordTrans_R(zhu_x[-1], zhu_y[-1], zuo_len, zuo_angle_mod)
    you_x, you_y = CoordTrans_R(zhu_x[-1], zhu_y[-1], you_len, you_angle_mod)

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
    plt.figure(2)
    plt.plot(zhu_x, zhu_y, color='red', linewidth=10 * zhu_diam)
    plt.plot(zuo_x, zuo_y, color='green', linewidth=10 * zuo_diam)
    plt.plot(you_x, you_y, color='blue', linewidth=10 * you_diam)
    # plt.xlim(-4,1)
    # plt.ylim(-2,3)
    plt.show()

    zhu = zhu_len_mod + zhu_angle_mod.tolist() + [zhu_diam]
    zuo = zuo_len + zuo_angle_mod.tolist() + [bif_expo]
    you = you_len + you_angle_mod.tolist() + [asy_ratio]
    entrance = [[zhu] + [zuo] + [you] + [min_tag]]
    return entrance


def draw_branch(branch):
    zhu = branch[0]
    zuo = branch[1]
    you = branch[2]
    min_tag = branch[3]
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
    zhu_x, zhu_y = CoordTrans_R(0, 0, zhu_len, zhu_angle)
    zuo_x, zuo_y = CoordTrans_R(0, 0, zuo_len, zuo_angle)
    you_x, you_y = CoordTrans_R(0, 0, you_len, you_angle)

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
    plt.figure(1)
    plt.plot(zhu_x, zhu_y, color='red', linewidth=10 * zhu_diam)
    plt.plot(zuo_x, zuo_y, color='green', linewidth=10 * zuo_diam)
    plt.plot(you_x, you_y, color='blue', linewidth=10 * you_diam)
    # plt.show()


if __name__ == "__main__":
    #
    Max_Part_len = 200
    Max_Diam = 70

    # 读取数据
    Net = ['Men_389','Men_546','Men_913']
    # NetType = ['Convergence']
    NetType = ['Bifurcation']

    for net in Net:
        for nt in NetType:

            data = np.load('../Data/Normalized Data/Coord&Diam_%s_%s.npy' % (net, nt),allow_pickle=True)
            feature = h5py.File('../Data/Normalized Data/k&lambda_%s_%s.mat' % (net, nt),mode='r')
            k_lambda = feature['Diam'][:]

            if net == 'Men_546':
                UnitRate = 2.88
            else:
                UnitRate = 1

            MainDiamMat = []
            for i in range(len(data)):
                Main = data[i][0]
                MainDiamMat.append(Main[-1])

            root_index = MainDiamMat.index(max(MainDiamMat))
            Main = data[root_index][0]
            Left = data[root_index][1]
            Right = data[root_index][2]

            Main_angle, Main_len = angle_len_norm(Main,UnitRate,Max_Part_len)
            Left_angle, Left_len = angle_len_norm(Left,UnitRate,Max_Part_len)
            Right_angle, Right_len = angle_len_norm(Right,UnitRate,Max_Part_len)

            Main_Diam = diam_norm(Main[-1],Max_Diam)
            bif_expo = k_lambda[3][root_index]
            asy_ratio = k_lambda[4][root_index]
            min_tag = k_lambda[5][root_index]

            insert_num = 32
            zhu = Main_len + Main_angle + [Main_Diam]
            zuo = Left_len + Left_angle + [bif_expo]
            you = Right_len + Right_angle + [asy_ratio]
            num_label = [0.] * insert_num
            num_label[int(min_tag)-1] = 1

            entrance = [[zhu] + [zuo] + [you] + [num_label]]
            draw_branch(entrance[0])
            entrance2 = entrance_mod(entrance[0])
            entrance3 = trans4210(entrance2[0])
            np.save("../Data/Normalized Data/Entrance_%s_%s.npy" % (net, nt), entrance3)
            print("Entrance_%s_%s.npy is saved" %(net, nt))

