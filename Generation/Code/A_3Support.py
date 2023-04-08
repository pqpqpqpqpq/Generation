"""
1-获取管径，分段长，确定用于归一化的数值
2-提取管径，存入.mat文件，用于计算K和lambda
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import os
from scipy import io


def angle_len(data, unitrate):
    """
    计算一个branch的每个segment的长度和角度信息，替换掉原来的坐标信息。
    input ；
        data，一个branch的数据
        unitrate，缩放比例
    return：
        part_angle_mat, 每个segment与x轴正半轴的夹角
        part_len_mat，每个segment的长度
    """
    x = data[0:len(data) // 2]
    y = data[len(data) // 2:-1]

    part_len_mat = []
    part_angle_mat = []

    for i in range(len(x) - 1):
        part_len = np.sqrt((x[i + 1] - x[i]) ** 2 + (y[i + 1] - y[i]) ** 2) * unitrate
        part_len_mat.append(part_len)

        part_angle = math.atan2(y[i + 1] - y[i], x[i + 1] - x[i])
        part_angle = part_angle / math.pi * 180
        if part_angle < 0:
            part_angle += 360
        part_angle_mat.append(part_angle)

    return part_angle_mat, part_len_mat


def Get_Jiajiao(Data1, Data2):
    """
    input:
        Data1,一个branch的数据
        Data2，另一个branch的数据
    return:
        math.acos(angel_cos)/math.pi*180，两个branch之间的夹角
    """
    x1 = Data1[0:len(Data1) // 2]
    y1 = Data1[len(Data1) // 2:-1]

    x2 = Data2[0:len(Data2) // 2]
    y2 = Data2[len(Data2) // 2:-1]

    seg1_x = x1[0] - x1[1]
    seg1_y = y1[0] - y1[1]

    seg2_x = x2[1] - x2[0]
    seg2_y = y2[1] - y2[0]

    angel_cos = (seg1_x * seg2_x + seg1_y * seg2_y) / (
            np.sqrt(seg1_x ** 2 + seg1_y ** 2) * np.sqrt(seg2_x ** 2 + seg2_y ** 2))
    return math.acos(angel_cos) / math.pi * 180


if __name__ == "__main__":
    # 读取数据
    Net = ['Men_389', 'Men_546', 'Men_913']
    # NetType = ['Convergence']
    NetType = ['Bifurcation']

    print('分析%s数据' % NetType[0])
    if not os.path.exists('../Support_Material/A_3Support/'):
        os.makedirs('../Support_Material/A_3Support/')

    All_len = []
    All_diam = []
    All_MainDiam = []
    All_MainLen = []

    Jiajiao_list = []
    zy_jiajiao = []
    for net in Net:
        for nt in NetType:

            data = np.load('../Data/Normalized Data/Coord&Diam_%s_%s.npy' % (net, nt), allow_pickle=True)
            DiamMat = []

            if net == 'Men_546':  # 546这个网络更小，这里保证三个网络在差不多的尺寸
                UnitRate = 2.88
            else:
                UnitRate = 1

            for i in range(len(data)):
                Main = data[i][0]
                Left = data[i][1]
                Right = data[i][2]

                Left_Jiajiao = Get_Jiajiao(Main, Left)
                Right_Jiajiao = Get_Jiajiao(Main, Right)
                zy_jiajiao.append(Get_Jiajiao(Left, Right))
                Jiajiao_list.append(abs(Left_Jiajiao - Right_Jiajiao) / 2)

                Main_angle, Main_len = angle_len(Main, UnitRate)
                Left_angle, Left_len = angle_len(Left, UnitRate)
                Right_angle, Right_len = angle_len(Right, UnitRate)

                Length = Main_len + Left_len + Right_len
                Diam = [Main[-1]] + [Left[-1]] + [Right[-1]]

                DiamMat.append(Diam)
                All_MainDiam.append(Main[-1])
                All_MainLen.append(sum(Main_len))

                for file in Length:
                    All_len.append(file)

                for file in Diam:
                    All_diam.append(file)
            io.savemat('../Data/Normalized Data/Diam_%s_%s.mat' % (net, nt), {'Diam': DiamMat})

    print('血管段子段最长：', max(All_len))
    print('血管段子段最短：', min(All_len))
    print('血管段子段长度统计特性：', np.mean(All_len), '±', np.std(All_len))

    plt.figure(1)
    plt.hist(All_len, list(range(0, 401, 10)))
    plt.title('%s_len_distribution' % (NetType[0]))
    plt.savefig('../Support_Material/A_3Support/%s_len_distribution.jpg' % (NetType[0]))

    print('血管段管径最大值：', max(All_diam))
    print('血管段管径最小值：', min(All_diam))
    print('血管段管径统计特性：', np.mean(All_diam), '±', np.std(All_diam))

    plt.figure(2)
    plt.title('Diam_distribution')
    plt.hist(All_diam, list(range(0, 101, 2)))
    plt.title('%s_diam_distribution' % (NetType[0]))
    plt.savefig('../Support_Material/A_3Support/%s_diam_distribution.jpg' % (NetType[0]))

    All_MainDiam.sort()
    l = len(All_MainDiam)
    print(All_MainDiam[l * 1 // 5], All_MainDiam[l * 2 // 5], All_MainDiam[l * 3 // 5], All_MainDiam[l * 4 // 5])

    All_MainLen.sort()
    l = len(All_MainLen)
    print(All_MainLen[0], All_MainLen[l * 1 // 3], All_MainLen[l * 2 // 3], All_MainLen[l - 1])
    print('最终选定的用于归一化的最大管径：70')
    print('最终选定的用于归一化的最大子段长度：200')

    plt.show()
