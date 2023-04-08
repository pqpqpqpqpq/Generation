"""
1-数据归一化
"""

import numpy as np
import math
import h5py


def angle_len_norm(data, unitrate, Max_Part_len):
    x = data[0:len(data) // 2]
    y = data[len(data) // 2:-1]

    part_len_mat = []
    part_angle_mat = []

    for i in range(len(x) - 1):
        part_len = np.sqrt((x[i + 1] - x[i]) ** 2 + (y[i + 1] - y[i]) ** 2) * unitrate
        if part_len > Max_Part_len:
            part_len = Max_Part_len
        part_len_mat.append(part_len)

        part_angle = math.atan2(y[i + 1] - y[i], x[i + 1] - x[i]) / math.pi * 180
        if part_angle < 0:
            part_angle += 360
        part_angle_mat.append(part_angle)

    part_len_mat = np.array(part_len_mat) / Max_Part_len
    part_angle_mat = np.array(part_angle_mat) / 360
    return part_angle_mat.tolist(), part_len_mat.tolist()


def diam_norm(D, Max_Diam):
    if D > Max_Diam:
        return 1
    else:
        return (D / Max_Diam)


if __name__ == "__main__":

    Max_Part_len = 200
    Max_Diam = 70

    # 读取数据
    Net = ['Men_389', 'Men_546', 'Men_913']
    NetType = ['Bifurcation', 'Convergence']

    All_len = []
    All_diam = []
    All_MainDiam = []

    seg_len = []

    for net in Net:
        for nt in NetType:

            data = np.load('../Data/Normalized Data/Coord&Diam_%s_%s.npy' % (net, nt), allow_pickle=True)
            feature = h5py.File('../Data/Normalized Data/k&lambda_%s_%s.mat' % (net, nt), mode='r')
            k_lambda = feature['Diam'][:]

            DiamMat = []

            if net == 'Men_546':
                UnitRate = 2.88
            else:
                UnitRate = 1

            NormData = []
            for i in range(len(data)):
                Main = data[i][0]
                Left = data[i][1]
                Right = data[i][2]

                Main_angle, Main_len = angle_len_norm(Main, UnitRate, Max_Part_len)  # 角度和分段长度归一化（分段长度最大值取200）
                Left_angle, Left_len = angle_len_norm(Left, UnitRate, Max_Part_len)
                Right_angle, Right_len = angle_len_norm(Right, UnitRate, Max_Part_len)

                Main_Diam = diam_norm(Main[-1], Max_Diam)
                bif_expo = k_lambda[3][i]
                asy_ratio = k_lambda[4][i]
                min_tag = k_lambda[5][i]

                seg_len.append(len(Main_len))
                seg_len.append(len(Left_len))
                seg_len.append(len(Right_len))

                insert_num = 32
                insert_Main_len = Main_len + [0.] * (insert_num - len(Main_len))
                insert_Left_len = Left_len + [0.] * (insert_num - len(Left_len))
                insert_Right_len = Right_len + [0.] * (insert_num - len(Right_len))
                insert_Main_angle = Main_angle + [0.] * (insert_num - len(Main_angle))
                insert_Left_angle = Left_angle + [0.] * (insert_num - len(Left_angle))
                insert_Right_angle = Right_angle + [0.] * (insert_num - len(Right_angle))
                insert_Main_diam = [Main_Diam] + [0.] * (insert_num - 1)
                insert_bif_expo = [bif_expo] + [0.] * (insert_num - 1)
                insert_asy_ratio = [asy_ratio] + [0.] * (insert_num - 1)

                insert_Main_diam[len(Main_len)] = 1
                insert_bif_expo[len(Left_len)] = 1
                insert_asy_ratio[len(Right_len)] = 1
                num_label = [0.] * insert_num
                num_label[int(min_tag) - 1] = 1

                insert = [insert_Main_len] + \
                         [insert_Left_len] + \
                         [insert_Right_len] + \
                         [insert_Main_angle] + \
                         [insert_Left_angle] + \
                         [insert_Right_angle] + \
                         [insert_Main_diam] + \
                         [insert_bif_expo] + \
                         [insert_asy_ratio] + \
                         [num_label]
                NormData.append(insert)
            finaldata = np.array((NormData))
            np.save("../Data/Normalized Data/Norm_%s_%s.npy" % (net, nt), finaldata)
            print("Norm_%s_%s.npy is saved" % (net, nt))
