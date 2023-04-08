"""
沈：
输入：真实的分叉数据
输出：分层的特征值
ps：
用于作为CV绘图的金标准
"""

from A_3Support import angle_len
import numpy as np
import pandas as pd
import math
import os

# 反转主分支坐标
def reverseList(zhu):
    zhu_x = zhu[0:len(zhu) // 2]
    zhu_y = zhu[len(zhu) // 2:(len(zhu) - 1)]
    diam = [zhu[-1]]
    zhu_x.reverse()
    zhu_y.reverse()
    zhu_reverse = zhu_x + zhu_y + diam
    return zhu_reverse


# 坐标转换
def CoordTrans(part_len, angle):
    Coord_x = [0]
    Coord_y = [0]
    for i in range(len(part_len)):
        rad = angle[i] * 2 * math.pi
        Coord_x.append(Coord_x[-1] + part_len[i] * math.cos(rad))
        Coord_y.append(Coord_y[-1] + part_len[i] * math.sin(rad))
    return Coord_x, Coord_y


# 计算卷曲度
def tort(part_len, angle):
    Coord_x, Coord_y = CoordTrans(part_len, angle)
    seg_len = np.sqrt((Coord_x[-1] - Coord_x[0]) ** 2 + (Coord_y[-1] - Coord_y[0]) ** 2)
    tort = sum(part_len) / seg_len
    return tort

def angle_len_norm(data, unitrate, Max_Part_len):
    x = data[0:len(data) // 2]
    y = data[len(data) // 2:-1]

    part_len_mat = []
    part_angle_mat = []

    for i in range(len(x) - 1):
        part_len = np.sqrt((x[i + 1] - x[i]) ** 2 + (y[i + 1] - y[i]) ** 2)*unitrate
        if part_len > Max_Part_len:
            part_len = Max_Part_len
        part_len_mat.append(part_len)

        part_angle = math.atan2(y[i + 1] - y[i], x[i + 1] - x[i])/ math.pi * 180
        if part_angle < 0:
            part_angle += 360
        part_angle_mat.append(part_angle)

    part_len_mat = np.array(part_len_mat)/Max_Part_len
    part_angle_mat = np.array(part_angle_mat)/360
    return part_angle_mat.tolist(), part_len_mat.tolist()


def get_feature(data):
    Max_Part_len = 200

    dataout = []
    if net == '546':
        UnitRate = 2.88
    else:
        UnitRate = 1

    main = data[0]
    left = data[1]
    right = data[2]

    # Main_angle, Main_len = angle_len(main, UnitRate)
    # Left_angle, Left_len = angle_len(left, UnitRate)
    # Right_angle, Right_len = angle_len(right, UnitRate)

    Main_angle, Main_len = angle_len_norm(main, UnitRate, Max_Part_len)  # 归一化
    Left_angle, Left_len = angle_len_norm(left, UnitRate, Max_Part_len)
    Right_angle, Right_len = angle_len_norm(right, UnitRate, Max_Part_len)

    a = abs(Left_angle[0] - Right_angle[0])
    if a < 180:
        angle = a
    else:
        angle = (360 - a)

    MainTort = tort(Main_len, Main_angle)
    LTort = tort(Left_len, Left_angle)
    RTort = tort(Right_len, Right_angle)

    MainLen = (sum(Main_len))
    LLen = (sum(Left_len))
    RLen = (sum(Right_len))

    main = main + [MainTort] + [angle] + [MainLen] + [net]
    left = left + [LTort] + [angle] + [LLen] + [net]
    right = right + [RTort] + [angle] + [RLen] + [net]

    dataout.append(main)
    dataout.append(left)
    dataout.append(right)

    return dataout


if __name__ == "__main__":
    Net = ['389', '546', '913']
    NetType,TreeType = 'Bifurcation',['Art']
    # NetType, TreeType = 'Convergence', ['Ven']

    """
    原始树数据的处理-这里一般来说跑一次就可以了。
    """
    # 处理原始数据
    # 挑选出只含有包含子分支的分叉,也就是说一个分叉的两个子分支,至少有一个匹配库里另一个分叉的主分支
    print('从训练数据中提取原始树的数据：')
    print('归一化之前的坐标数据')
    for net in Net:
        final = np.load('../Data/Normalized Data/Coord&Diam_Men_%s_%s.npy' % (net, NetType), allow_pickle=True)
        print('处理数据：','../Data/Normalized Data/Coord&Diam_Men_%s_%s.npy' % (net, NetType))
        # 获取入口分叉
        MainDiamMat = []
        for i in range(len(final)):
            Main = final[i][0]  # 获取主分支
            MainDiamMat.append(Main[-1])  # 获取主分支管径
        root_index = MainDiamMat.index(max(MainDiamMat))
        Entrance = final[root_index]

        allData = get_feature(Entrance)
        allData[0] = allData[0] + [1]
        allData[1] = allData[1] + [2]
        allData[2] = allData[2] + [2]

        dataout = allData[1:]

        levelout = []  # 定义数据
        while (len(dataout) != 0):
            for i in range(len(dataout)):

                for j in range(len(final)):
                    if dataout[i][0:-5] == reverseList(final[j][0]):
                        next_level = get_feature(final[j])
                        next_level[1] = next_level[1] + [dataout[i][-1] + 1]
                        next_level[2] = next_level[2] + [dataout[i][-1] + 1]
                        levelout.append(next_level[1])
                        levelout.append(next_level[2])
                        # print(levelout)
                        allData.append(next_level[1])
                        allData.append(next_level[2])
                        # print(allData)
                        print('完成第%s层匹配' % [dataout[i][-1] + 1])
            dataout = levelout
            levelout = []  # 清空数据

        allFeature = []
        for i in range(len(allData)):
            data = allData[i]
            feature = data[-6:]
            allFeature.append(feature)

        allFeature = np.array(allFeature)
        allFeature = pd.DataFrame(allFeature)  # 转换数据类型为pandas

        if not os.path.exists('../Support_Material/D_3OriTreetoExcel/%s/' % NetType):
            os.makedirs('../Support_Material/D_3OriTreetoExcel/%s/' % NetType)

        # 存数据画图
        np.save("../Support_Material/D_3OriTreetoExcel/%s/%s.npy" % (NetType, net), allData)
        allData = np.array(allData)
        allData = pd.DataFrame(allData)
        writer0 = pd.ExcelWriter('../Support_Material/D_3OriTreetoExcel/%s/all_data_%s.xlsx' % (NetType, net))
        allData.to_excel(writer0, sheet_name='DataPerGen', index=True)
        writer0.save()
        writer0.close()

        headers = ['Diam','tort','angle', 'len', 'index', 'generation']
        writer = pd.ExcelWriter('../Support_Material/D_3OriTreetoExcel/%s/%s.xlsx' % (NetType, net))
        # header参数表示列的名称，index表示行的标签
        # 没办法编辑index
        allFeature.to_excel(writer, sheet_name='2', header=headers, index=False)
        writer.save()
        writer.close()
