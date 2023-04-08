"""
函数库，主要有以下函数，该函数上的为其配套函数
1. CrashBranchRemove：碰撞自检函数，移除血管段发生碰撞的分叉
1+1. AbnormalEliminate: 去除某一分叉中连续两个子分叉角度变化过大的以及左右分叉夹角过小的
2. GrowSeqSort：将所有分叉的左右子分叉按管径大小排列，分叉对应的Seg和Coord按同样顺序排列
3. get_coord_seg：由训练数据/生成数据获取真实比例下的Coord（坐标）和Seg（管径/管长/卷曲度）
4. get_condition：根据输入的管径和管长获取条件
5. angle_determine：基于SALFD确定血管分叉的生长方向
6. Cal_Error: 计算生成的分叉（data_son）的主血管段与待生长的血管段（data_mom）之间的加权误差，并按误差从小到大排列
7. Gen_Rotate：根据angle_determine中确定的血管分叉生成方向调整生成的branch的角度以方便拼接
8. CrashDetection：判断新生成的分叉是否与已有的分叉相交
9. SegExtract: 提取branch中左分叉和右分叉的【管径，管长，卷曲度】
"""

import numpy as np
import operator
import cv2
import math
import matplotlib.pyplot as plt
import random

from scipy import optimize
from B_1DSNGAN import CoordTrans
from C_GetEntrance import CoordTrans_R


def tort(part_len, angle):
    Coord_x, Coord_y = CoordTrans(part_len, angle)
    seg_len = np.sqrt((Coord_x[-1] - Coord_x[0]) ** 2 + (Coord_y[-1] - Coord_y[0]) ** 2)
    tort = sum(part_len) / seg_len
    return tort


def reverseList(zhu):
    zhu_x = zhu[0:len(zhu) // 2]
    zhu_y = zhu[len(zhu) // 2:(len(zhu) - 1)]
    diam = [zhu[-1]]
    zhu_x.reverse()
    zhu_y.reverse()
    zhu_reverse = zhu_x + zhu_y + diam
    return zhu_reverse


def cross(a, b, c):  # 跨立实验
    x1 = b[0] - a[0]
    y1 = b[1] - a[1]
    x2 = c[0] - a[0]
    y2 = c[1] - a[1]
    return x1 * y2 - x2 * y1


def IsIntersec(a1, a2, b1, b2):  # 判断两线段是否相交
    # 快速排斥，以l1、l2为对角线的矩形必相交，否则两线段不相交
    if (max(a1[0], a2[0]) >= min(b1[0], b2[0])  # 矩形1最右端大于矩形2最左端
            and max(b1[0], b2[0]) >= min(a1[0], a2[0])  # 矩形2最右端大于矩形最左端
            and max(a1[1], a2[1]) >= min(b1[1], b2[1])  # 矩形1最高端大于矩形最低端
            and max(b1[1], b2[1]) >= min(a1[1], a2[1])):  # 矩形2最高端大于矩形最低端
        # 若通过快速排斥则进行跨立实验
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

    gen_well_data = np.array(gen_well_data)
    return gen_well_data


def anglelistdetec(data):  # 检测某一分叉中连续两个子分叉是否有变化超过angle_lim的
    flag = True
    angle_lim = 60
    for i in range(len(data) - 1):
        a = abs(data[i + 1] - data[i]) * 360
        b = 360 - abs(data[i + 1] - data[i]) * 360
        if a > angle_lim and b > angle_lim:
            flag = False
            break
    return flag


# 去除某一分叉中连续两个子分叉角度变化过大的以及左右分叉夹角过小的
def AbnormalEliminate(data):
    gen_well_data = []
    for i in range(len(data)):
        img = data[i]
        zuo = img[1]
        you = img[2]

        zuo_angle = zuo[len(zuo) // 2:(len(zuo) - 1)]
        you_angle = you[len(you) // 2:(len(you) - 1)]

        a = abs(zuo_angle[0] - you_angle[0]) * 360 < 20
        b = (360 - abs(zuo_angle[0] - you_angle[0]) * 360) < 20

        flag1 = True
        if a or b:
            flag1 = False

        c = abs(zuo_angle[0] - you_angle[0]) * 360 > 125
        d = (360 - abs(zuo_angle[0] - you_angle[0]) * 360) > 125
        flag2 = True
        if c and d:
            flag2 = False

        if flag1 and flag2 and anglelistdetec(zuo_angle) and anglelistdetec(you_angle):
            gen_well_data.append(data[i])

    gen_well_data = np.array(gen_well_data)
    return gen_well_data


# 将所有分叉的左右子分叉按管径大小排列，分叉对应的Seg和Coord按同样顺序排列
def GrowSeqSort(Tree_Data, Coord_Data):
    Seg_Mom_List = []
    Coord_Mom_List = []
    Diam_Mom_List = []
    Seg_Index = 0

    MaxLevel = Tree_Data[-1][-1]
    for i in range(len(Tree_Data)):
        if Tree_Data[i][-1] == MaxLevel:
            Seg_Mom_List.append(Tree_Data[i])
            Coord_Mom_List.append(Coord_Data[i])
            Diam_Mom_List.append([Tree_Data[i][3], Seg_Index])
            Seg_Index += 1
    Diam_Mom_List.sort(key=operator.itemgetter(0), reverse=True)  # 按管径降序排列

    Index_List = []
    for i in range(len(Diam_Mom_List)):
        Index_List.append(Diam_Mom_List[i][-1])

    return Seg_Mom_List, Coord_Mom_List, Index_List


# 由训练数据/生成数据获取真实比例下的Coord（坐标）和Seg（管径/管长/卷曲度）
def get_coord_seg(data):
    data = data.tolist()
    zhu_len = data[0]
    zuo_len = data[1]
    you_len = data[2]
    zhu_angle = data[3]
    zuo_angle = data[4]
    you_angle = data[5]
    zhu_diam = data[6]
    bif_expo = data[7]
    asy_ratio = data[8]
    min_tag = data[9]

    zhu_len = zhu_len[0:zhu_diam[1:].index(max(zhu_diam[1:])) + 1]
    zuo_len = zuo_len[0:bif_expo[1:].index(max(bif_expo[1:])) + 1]
    you_len = you_len[0:asy_ratio[1:].index(max(asy_ratio[1:])) + 1]
    zhu_angle = zhu_angle[0:zhu_diam[1:].index(max(zhu_diam[1:])) + 1]
    zuo_angle = zuo_angle[0:bif_expo[1:].index(max(bif_expo[1:])) + 1]
    you_angle = you_angle[0:asy_ratio[1:].index(max(asy_ratio[1:])) + 1]

    a = abs(zuo_angle[0] - you_angle[0]) * 360
    if a < 180:
        angle = a
    else:
        angle = 360 - a

    tag = min_tag[0:2].index(max(min_tag[0:2]))

    zhu_x, zhu_y = CoordTrans_R(0, 0, np.array(zhu_len) * 200, zhu_angle)
    zuo_x, zuo_y = CoordTrans_R(zhu_x[-1], zhu_y[-1], np.array(zuo_len) * 200, zuo_angle)
    you_x, you_y = CoordTrans_R(zhu_x[-1], zhu_y[-1], np.array(you_len) * 200, you_angle)

    # 原始分叉指数获得及子血管段管径计算
    BE = bif_expo[0] * 4.5 + 1.5
    major_diam = zhu_diam[0] / (asy_ratio[0] ** BE + 1) ** (1.0 / BE)
    minor_diam = major_diam * asy_ratio[0]

    if tag == 0:
        zuo_diam = minor_diam
        you_diam = major_diam
    elif tag == 1:
        zuo_diam = major_diam
        you_diam = minor_diam

    zhu_coord = [zhu_x, zhu_y]
    zuo_coord = [zuo_x, zuo_y]
    you_coord = [you_x, you_y]

    zhu_tort = tort(zhu_len, zhu_angle)
    zuo_tort = tort(zuo_len, zuo_angle)
    you_tort = tort(you_len, you_angle)

    zhu_seg_len = sum(np.array(zhu_len) * 200)
    zuo_seg_len = sum(np.array(zuo_len) * 200)
    you_seg_len = sum(np.array(you_len) * 200)

    # [Diam, Length, Tortuosity,k,lambda,angel]
    zhu_seg = [zhu_diam[0] * 70, zhu_seg_len, zhu_tort, BE, asy_ratio[0], angle]
    zuo_seg = [zuo_diam * 70, zuo_seg_len, zuo_tort, BE, asy_ratio[0], angle]
    you_seg = [you_diam * 70, you_seg_len, you_tort, BE, asy_ratio[0], angle]

    return zhu_coord, zuo_coord, you_coord, zhu_seg, zuo_seg, you_seg


# 根据输入的管径和管长获取条件
def get_condition(data, vestype='ven'):
    if vestype == 'ven':
        # diam_list = [0.0, 15.0, 21.0, 29.62, 38.5, 70.0]
        # len_list = [0.0, 150.0, 330.0, 2000.0]
        diam_list = [0.0, 15.0, 21.0, 29.62, 38.5, 85.0]
        len_list = [0.0, 149.07541246920727, 330.5998514199406, 2000]
    elif vestype == 'art':
        # diam_list = [0.0, 10.0, 13.0, 16.0, 20.0, 70.0]
        # len_list = [0.0, 140.7, 384.2, 2000.0]
        diam_list = [0.0, 10.0, 13.0, 16.0, 20.0, 85.0]
        len_list = [0.0, 140.69579269747607, 384.1959013618342, 2000]

    main_diam = data[0]
    main_len = data[1]

    condition = np.zeros((2,))
    for j in range(len(diam_list) - 1):
        if diam_list[j] < main_diam <= diam_list[j + 1]:
            condition[0] = -1 + j / 2
    for j in range(len(len_list) - 1):
        if len_list[j] < main_len <= len_list[j + 1]:
            condition[1] = -1 + j

    return condition


def AngleList2360(angle_list):  # 将角度列表内的数值全部调整到0~360范围内
    for i in range(len(angle_list)):
        if angle_list[i] < 0:
            angle_list[i] += 360
        elif angle_list[i] >= 360:
            angle_list[i] -= 360
    return angle_list


def Angle2360(angle):  # 将角度调整到0~360范围内
    if angle < 0:
        angle += 360
    elif angle >= 360:
        angle -= 360
    return angle


def ROI_save(folder, Tree_Data, Coord_Data, Coord_Mom, dpi, win_width):  # 保存以血管段端点为中心的正方形区域内图形
    start_x = Coord_Mom[0][-1]
    start_y = Coord_Mom[1][-1]

    umpp = 4  # um per pixel, 通过设置该参数调整像素点和实际长度（um）的比例关系(与dpi的取值有关)
    fig = plt.figure(
        figsize=(win_width / (umpp * dpi), win_width / (umpp * dpi)))  # 绘图保存的图片大小为win_width/umpp * win_width/umpp
    ax = fig.add_axes([0, 0, 1, 1])

    for j in range(len(Tree_Data)):
        # linewidth单位为points，1 point = 1/72 inch, 1 inch = dpi pixels, 1 pixel = umpp um
        # 则管径为D um的血管段，画图时对应的linewidth为72 /(umpp*dpi) * D
        ax.plot(Coord_Data[j][0], Coord_Data[j][1], 'k-', linewidth=72 / (umpp * dpi) * Tree_Data[j][3])

    ax.axis('off')
    ax.axis([start_x - win_width / 2, start_x + win_width / 2, start_y - win_width / 2, start_y + win_width / 2])

    fig_name = folder + 'ROI.tiff'
    plt.savefig(fig_name, dpi=dpi, pil_kwargs={"compression": "tiff_lzw"})
    plt.close()
    return fig_name


def fun_fit(x, A, B):  # 用于拟合斜率的函数
    return A * x + B


def Cal_FD(img):  # 计算分形维数
    m, n = img.shape
    box_size_tmp = 2 ** np.arange(0, np.log2(m))
    box_size_list = [int(i) for i in box_size_tmp]
    box_num_list = np.zeros((len(box_size_list),))
    for i in range(len(box_size_list)):
        start = list(range(0, m, box_size_list[i]))
        box_num = 0
        for j in range(len(start)):
            for k in range(len(start)):
                box = img[start[j]:start[j] + box_size_list[i], start[k]:start[k] + box_size_list[i]]
                if sum(sum(box)) > 0:
                    box_num += 1
        box_num_list[i] = box_num
    A, B = optimize.curve_fit(fun_fit, np.log(box_size_list), np.log(box_num_list))[0]
    return abs(A)


# 确定血管分叉的生长方向
def angle_determine(folder, Tree_Data, Coord_Data, Coord_Mom):
    dpi = 600
    win_width = 1024  # ROI所在窗的的大小，单位为um，2的整数次幂，方便后面计算
    img_path = ROI_save(folder, Tree_Data, Coord_Data, Coord_Mom, dpi, win_width)

    # 图像二值化
    ROI_img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(ROI_img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 127, 1, cv2.THRESH_BINARY_INV)

    w, l = binary_img.shape
    win = int(w / 8)
    step = round(win / 6)
    start_list = np.round(np.linspace(0, w - win, int((w - win) / step)))

    # 计算ROI的加权平均局部分形维数
    FD_Mat = np.zeros((len(start_list), len(start_list)))
    A = np.zeros((w, l))
    B = np.zeros((w, l))

    for i in range(len(start_list)):
        for j in range(len(start_list)):
            a = int(start_list[i])
            b = int(start_list[j])
            sub_img = binary_img[a:a + win, b:b + win]
            if sum(sum(sub_img)) == 0:  # 没像素点
                FD_Mat[i, j] = 0
                A[a:a + win, b:b + win] += 0
                B[a:a + win, b:b + win] += 1
            else:
                FD_Mat[i, j] = Cal_FD(sub_img)
                A[a:a + win, b:b + win] += Cal_FD(sub_img)
                B[a:a + win, b:b + win] += 1

    SALFD = A / B

    # 计算SALFD的质心
    sum_x = np.sum(SALFD, axis=0)
    sum_y = np.sum(SALFD, axis=1)

    weight_x = np.sum(sum_x * np.arange(len(sum_x))) / np.sum(sum_x)
    weight_y = np.sum(sum_y * np.arange(len(sum_y))) / np.sum(sum_y)

    centroid_dir = [weight_x - l / 2, w / 2 - weight_y]  # 图形的纵坐标和常用坐标系的纵坐标是反过来的，所以后面一项是反过来的
    centroid_angle = math.atan2(centroid_dir[1], centroid_dir[0]) / math.pi * 180  # centroid_angle范围[-180,180]

    # 确定初步的生长方向
    end_dir = [Coord_Mom[0][-1] - Coord_Mom[0][-2], Coord_Mom[1][-1] - Coord_Mom[1][-2]]
    end_angle = math.atan2(end_dir[1], end_dir[0]) / math.pi * 180  # end_angle范围[-180,180]

    # 画出SALFD图及质心方向
    plt.matshow(SALFD)

    # plt.axis('off')
    plt.colorbar()
    fig_name = img_path[0:-8] + 'SALFD.tiff'
    plt.savefig(fig_name, dpi=600, pil_kwargs={"compression": "tiff_lzw"})

    plt.annotate('', xy=(l / 2, w / 2), xytext=(weight_x, weight_y),
                 arrowprops=dict(arrowstyle="->", color='r', linewidth=1.5))
    fig_name = img_path[0:-8] + 'ZCD.tiff'
    plt.savefig(fig_name, dpi=600, pil_kwargs={"compression": "tiff_lzw"})

    plt.close()

    return centroid_angle + 180, Angle2360(end_angle)


# 计算生成的分叉（data_son）的主血管段与待生长的血管段（data_son）之间的加权误差，并按误差从小到大排列
def Cal_Error(data_mom, data_son, th=0.1):
    error_mat = []
    for i in range(len(data_son)):
        zhu = data_son[i][0]
        zhu_len = zhu[0:len(zhu) // 2]
        zhu_angle = zhu[len(zhu) // 2:(len(zhu) - 1)]

        # 坐标转换
        zhu_seg_diam = zhu[-1] * 70
        zhu_seg_len = sum(np.array(zhu_len) * 200)
        zhu_seg_tort = tort(zhu_len, zhu_angle)

        error = 0.6 * abs(data_mom[0] - zhu_seg_diam) / data_mom[0] + 0.3 * abs(data_mom[1] - zhu_seg_len) / data_mom[1] \
                + 0.1 * abs(data_mom[2] - zhu_seg_tort) / data_mom[2]
        if error < th:
            error_mat.append([error, i])

    error_mat.sort(key=operator.itemgetter(0))
    return error_mat


def angle_mod(angle_list, centroid_direction_rev, end_angle, flag):  # 用质心的方向去提供一个偏量
    new_angle = []

    if flag:
        a = 0.3
        b = 0.1
    else:
        a = 0.3
        b = 0

    # if flag:
    #     a = 0
    #     b = 0
    # else:
    #     a = 0
    #     b = 0

    for i in angle_list:
        angle_tan = np.tan(i * math.pi / 180)
        if 90 < i <= 270:
            tmp_xl = [-1 / np.sqrt(angle_tan ** 2 + 1), -angle_tan / np.sqrt(angle_tan ** 2 + 1)]
        else:
            tmp_xl = [1 / np.sqrt(angle_tan ** 2 + 1), angle_tan / np.sqrt(angle_tan ** 2 + 1)]

        grow_tan = np.tan(centroid_direction_rev * math.pi / 180)
        if 90 < centroid_direction_rev <= 270:
            grow_xl = [-1 / np.sqrt(grow_tan ** 2 + 1), -grow_tan / np.sqrt(grow_tan ** 2 + 1)]
        else:
            grow_xl = [1 / np.sqrt(grow_tan ** 2 + 1), grow_tan / np.sqrt(grow_tan ** 2 + 1)]

        end_tan = np.tan(end_angle * math.pi / 180)
        if 90 < end_angle <= 270:
            end_xl = [-1 / np.sqrt(end_tan ** 2 + 1), -end_tan / np.sqrt(end_tan ** 2 + 1)]
        else:
            end_xl = [1 / np.sqrt(end_tan ** 2 + 1), end_tan / np.sqrt(end_tan ** 2 + 1)]

        new_xl = [(1 - a - b) * tmp_xl[0] + a * grow_xl[0] + b * end_xl[0],
                  (1 - a - b) * tmp_xl[1] + a * grow_xl[1] + b * end_xl[1]]

        new_angle.append(math.atan2(new_xl[1], new_xl[0]) / math.pi * 180)
    return new_angle


# 根据grow_direction调整branch
def Gen_Rotate(branch, centroid_direction_rev, end_angle):
    zuo = branch[1]
    you = branch[2]
    min_tag = branch[3]

    zuo_len = zuo[0:len(zuo) // 2]
    you_len = you[0:len(you) // 2]
    zuo_angle = zuo[len(zuo) // 2:(len(zuo) - 1)]
    you_angle = you[len(you) // 2:(len(you) - 1)]
    tag = min_tag[0:2].index(max(min_tag[0:2]))

    # 坐标转换
    zuo_x, zuo_y = CoordTrans_R(0, 0, zuo_len, zuo_angle)
    you_x, you_y = CoordTrans_R(0, 0, you_len, you_angle)

    angle_cos = (zuo_x[1] * you_x[1] + zuo_y[1] * you_y[1]) / (
            np.sqrt(zuo_x[1] ** 2 + zuo_y[1] ** 2) * np.sqrt(you_x[1] ** 2 + you_y[1] ** 2))
    jiajiao = math.acos(angle_cos) / math.pi * 180

    end_angle = Angle2360(end_angle + random.randint(-3, 3))

    if abs(max([zuo_angle[0], you_angle[0]]) * 360 - jiajiao / 2 - (
            zuo_angle[0] * 360 + you_angle[0] * 360) / 2) < 1:

        angle_adjust = end_angle - (zuo_angle[0] * 360 + you_angle[0] * 360) / 2
    else:
        branch = (zuo_angle[0] * 360 + you_angle[0] * 360 - 360) / 2
        if branch < 0:
            angle_adjust = end_angle - (branch + 360)
        else:
            angle_adjust = end_angle - branch

    zuo_angle_adjust = AngleList2360(np.array(zuo_angle) * 360 + angle_adjust)
    you_angle_adjust = AngleList2360(np.array(you_angle) * 360 + angle_adjust)

    if tag == 0:
        zuo_angle_mod = AngleList2360(
            np.array(angle_mod(zuo_angle_adjust, centroid_direction_rev, end_angle, False))) / 360
        you_angle_mod = AngleList2360(
            np.array(angle_mod(you_angle_adjust, centroid_direction_rev, end_angle, True))) / 360
    elif tag == 1:
        zuo_angle_mod = AngleList2360(
            np.array(angle_mod(zuo_angle_adjust, centroid_direction_rev, end_angle, True))) / 360
        you_angle_mod = AngleList2360(
            np.array(angle_mod(you_angle_adjust, centroid_direction_rev, end_angle, False))) / 360

    return zuo_len, you_len, zuo_angle_mod, you_angle_mod


# 判断生成的分叉是否与已有的分叉相交
def CrashDetection(Coord, Coord_branch):
    Coord_all_list = []
    for i in range(len(Coord)):
        Coord_x = Coord[i][0]
        Coord_y = Coord[i][1]
        for j in range(len(Coord_x) - 1):
            Coord_all_list.append([[Coord_x[j], Coord_y[j]], [Coord_x[j + 1], Coord_y[j + 1]]])

    Coord_branch_list = []
    Coord_branch_x = Coord_branch[0]
    Coord_branch_y = Coord_branch[1]
    for i in range(len(Coord_branch_x) - 1):
        Coord_branch_list.append(
            [[Coord_branch_x[i], Coord_branch_y[i]], [Coord_branch_x[i + 1], Coord_branch_y[i + 1]]])

    BoolenList = []
    for i in range(len(Coord_branch_list)):
        for j in range(len(Coord_all_list)):
            # 首先剔除端点相同的血管段
            if Coord_branch_list[i][0] == Coord_all_list[j][1]:
                continue
            # 判断是否相交
            Boolen = IsIntersec(Coord_branch_list[i][0], Coord_branch_list[i][1], Coord_all_list[j][0],
                                Coord_all_list[j][1])
            BoolenList.append(Boolen)

    if True not in BoolenList:
        return True
    else:
        return False


# 提取branch中左分叉和右分叉的【管径，管长，卷曲度】
def SegExtract(branch, zhu_diam):
    zuo = branch[1]
    you = branch[2]
    min_tag = branch[3]

    bif_expo = zuo[-1]
    asy_ratio = you[-1] + 0.04

    if zhu_diam > 35 and random.random() < 0.5:
        tag = 1 - min_tag[0:2].index(max(min_tag[0:2]))
    else:
        tag = min_tag[0:2].index(max(min_tag[0:2]))

    zuo_len = zuo[0:len(zuo) // 2]
    zuo_angle = zuo[len(zuo) // 2:(len(zuo) - 1)]
    you_len = you[0:len(you) // 2]
    you_angle = you[len(you) // 2:(len(you) - 1)]

    a = abs(zuo_angle[0] - you_angle[0]) * 360
    if a < 180:
        angle = a
    else:
        angle = 360 - a

    # 坐标转换
    zuo_seg_len = sum(np.array(zuo_len) * 200)
    zuo_seg_tort = tort(zuo_len, zuo_angle)

    you_seg_len = sum(np.array(you_len) * 200)
    you_seg_tort = tort(you_len, you_angle)

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

    # [Diam, Length, Tortuosity,k,lambda,angel]
    zuo_seg = [zuo_diam, zuo_seg_len, zuo_seg_tort, BE, asy_ratio, angle]
    you_seg = [you_diam, you_seg_len, you_seg_tort, BE, asy_ratio, angle]

    return zuo_seg, you_seg


if __name__ == '__main__':
    Data = np.load('../Data/Normalized Data/Norm_Men_546_Convergence.npy')
    print(1)
