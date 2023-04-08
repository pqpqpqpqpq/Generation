"""
1-打标签
2-扩充数据 convergence-(408到29376)
         bifurcation-(382到27504）
"""

import numpy as np


# 单独将坐标点提取出来，返回一个列表（分叉）
def select_axis(data):
    branch = []
    data = data.tolist()
    for i in range(len(data)):
        zhu_len = data[i][0]
        zuo_len = data[i][1]
        you_len = data[i][2]
        zhu_angle = data[i][3]
        zuo_angle = data[i][4]
        you_angle = data[i][5]
        zhu_diam = data[i][6]
        bif_expo = data[i][7]
        asy_ratio = data[i][8]
        min_tag = data[i][9]

        zhu_len = zhu_len[0:zhu_diam[1:].index(max(zhu_diam[1:])) + 1]
        zuo_len = zuo_len[0:bif_expo[1:].index(max(bif_expo[1:])) + 1]
        you_len = you_len[0:asy_ratio[1:].index(max(asy_ratio[1:])) + 1]
        zhu_angle = zhu_angle[0:zhu_diam[1:].index(max(zhu_diam[1:])) + 1]
        zuo_angle = zuo_angle[0:bif_expo[1:].index(max(bif_expo[1:])) + 1]
        you_angle = you_angle[0:asy_ratio[1:].index(max(asy_ratio[1:])) + 1]
        zhu = zhu_len + zhu_angle + [zhu_diam[0]]
        zuo = zuo_len + zuo_angle + [bif_expo[0]]
        you = you_len + you_angle + [asy_ratio[0]]

        fencha = [zhu] + [zuo] + [you] + [min_tag]
        branch.append(fencha)
    return branch


# 旋转数据
def rotate(angle, ori_angle):
    rotate_angle = []
    for i in range(len(ori_angle)):
        tmp = ori_angle[i] + angle / 360
        if tmp > 1:
            rotate_angle.append(tmp - 1)
        else:
            rotate_angle.append(tmp)
    return rotate_angle


# 旋转扩增数据
def enhance(Data, rotate_angle):
    rotatedata = []
    for i in range(len(Data)):
        zhu = Data[i][0]
        zuo = Data[i][1]
        you = Data[i][2]
        min_tag = Data[i][3]
        zhu_diam = [zhu[-1]]
        bif_expo = [zuo[-1]]
        asy_ratio = [you[-1]]
        zhu_len = zhu[0:len(zhu) // 2]
        zuo_len = zuo[0:len(zuo) // 2]
        you_len = you[0:len(you) // 2]
        zhu_angle = zhu[len(zhu) // 2:(len(zhu) - 1)]
        zuo_angle = zuo[len(zuo) // 2:(len(zuo) - 1)]
        you_angle = you[len(you) // 2:(len(you) - 1)]

        # 在这里分类，不一定非要360度
        for j in range(0, 360, rotate_angle):
            zhu_rotate = rotate(j, zhu_angle)
            zuo_rotate = rotate(j, zuo_angle)
            you_rotate = rotate(j, you_angle)

            insert_num = 32
            insert_zhu_len = zhu_len + [0.] * (insert_num - len(zhu_len))  # 在加号前加#号，把前面的插值加上,即为插值
            insert_zuo_len = zuo_len + [0.] * (insert_num - len(zuo_len))
            insert_you_len = you_len + [0.] * (insert_num - len(you_len))
            insert_zhu_angle = zhu_rotate + [0.] * (insert_num - len(zhu_rotate))
            insert_zuo_angle = zuo_rotate + [0.] * (insert_num - len(zuo_rotate))
            insert_you_angle = you_rotate + [0.] * (insert_num - len(you_rotate))
            insert_zhu_diam = zhu_diam + [0.] * (insert_num - len(zhu_diam))
            insert_bif_expo = bif_expo + [0.] * (insert_num - len(bif_expo))
            insert_asy_ratio = asy_ratio + [0.] * (insert_num - len(asy_ratio))
            insert_zhu_diam[len(zhu_rotate)] = 1
            insert_bif_expo[len(zuo_rotate)] = 1
            insert_asy_ratio[len(you_rotate)] = 1

            insert = [insert_zhu_len] + \
                     [insert_zuo_len] + \
                     [insert_you_len] + \
                     [insert_zhu_angle] + \
                     [insert_zuo_angle] + \
                     [insert_you_angle] + \
                     [insert_zhu_diam] + \
                     [insert_bif_expo] + \
                     [insert_asy_ratio] + \
                     [min_tag]

            rotatedata.append(insert)
    enhancedata = np.array(rotatedata)
    return enhancedata


# 获取条件
def get_condition(data, vestype='ven'):
    data_condition = []
    if vestype == 'ven':
        diam_list = [0.0, 15.0, 21.0, 29.62, 38.5, 85.0]
        len_list = [0.0, 149.07541246920727, 330.5998514199406, 2000]
    elif vestype == 'art':
        diam_list = [0.0, 10.0, 13.0, 16.0, 20.0, 85.0]
        len_list = [0.0, 140.69579269747607, 384.1959013618342, 2000]

    for i in range(len(data)):
        main_diam = data[i][6][0] * 70
        main_len = sum(data[i][0]) * 200
        condition = np.zeros((2,))
        for j in range(len(diam_list) - 1):
            if diam_list[j] < main_diam <= diam_list[j + 1]:
                condition[0] = -1 + j / 2

        for j in range(len(len_list) - 1):
            if len_list[j] < main_len <= len_list[j + 1]:
                condition[1] = -1 + j

        data_condition.append([data[i], condition])
    return data_condition


if __name__ == "__main__":
    # 读取数据
    Net = ['Men_389', 'Men_546', 'Men_913']
    NetType = ['Convergence', 'art']
    # NetType = ['Bifurcation','ven']
    print('拓展数据：%s' % NetType[0])

    combined_data = []
    for net in Net:
        Data = np.load("../Data/Normalized Data/Norm_%s_%s.npy" % (net, NetType[0]))
        for i in range(len(Data)):
            combined_data.append(Data[i])

    # 数据扩增
    combined_data = np.array(combined_data)
    print("Original Data Shape: ", combined_data.shape)
    tmp = select_axis(combined_data)
    enhancedata = enhance(tmp, 5)  # 数据扩增
    print("Enhanced Data Shape: ", enhancedata.shape)

    # 获取条件
    data_condition = get_condition(enhancedata, NetType[1])
    data_condition = np.array(data_condition)
    print("data_condition Shape: ", data_condition.shape)

    # 数据保存
    np.save("../Data/Normalized Data/%s_Data&Condition.npy" % NetType[0], data_condition)
