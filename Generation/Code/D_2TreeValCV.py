'''
比较真实树和生成树各指标的CV
'''

import numpy as np
from scipy import stats
from A_6DataEnhance import select_axis
from C_Func import tort
from scipy import io
import pandas as pd


def get_ori_CV(Net, NetType):
    # 真实血管树
    Diam_CV1 = []
    Len_CV1 = []
    Tort_CV1 = []
    K_CV1 = []
    lambda_CV1 = []
    angle_CV1 = []
    # all_ori = []
    for net in Net:

        Diam_mat = []
        Len_mat = []
        Tort_mat = []
        K_mat = []
        lambda_mat = []
        angle_mat = []

        # 沈：三个网络的均方数据

        for nt in NetType:
            DataSet = np.load("../Data/Normalized Data/Norm_Men_%s_%s.npy" % (net, nt))
            print('加载真实数据：', "../Data/Normalized Data/Norm_Men_%s_%s.npy" % (net, nt))
            DataSet = select_axis(DataSet)

            for i in range(len(DataSet)):
                Data = DataSet[i]

                zhu = Data[0]
                zuo = Data[1]
                you = Data[2]
                min_tag = Data[3]

                zhu_diam = zhu[-1]
                bif_expo = zuo[-1]
                asy_ratio = you[-1]
                zhu_len = zhu[0:len(zhu) // 2]
                zhu_angle = zhu[len(zhu) // 2:(len(zhu) - 1)]

                zuo_len = zuo[0:len(zuo) // 2]
                zuo_angle = zuo[len(zuo) // 2:(len(zuo) - 1)]

                you_len = zuo[0:len(you) // 2]
                you_angle = you[len(you) // 2:(len(you) - 1)]

                tag = min_tag[0:2].index(max(min_tag[0:2]))

                BE = bif_expo * 4.5 + 1.5
                major_diam = zhu_diam / (asy_ratio ** BE + 1) ** (1.0 / BE)
                minor_diam = major_diam * asy_ratio

                a = abs(zuo_angle[0] - you_angle[0]) * 360
                if a < 180:
                    angle_mat.append(a)
                else:
                    angle_mat.append(360 - a)

                Diam_mat.append(major_diam * 70)
                Diam_mat.append(minor_diam * 70)
                Len_mat.append(sum(zuo_len) * 200)
                Len_mat.append(sum(you_len) * 200)
                Tort_mat.append(tort(zuo_len, zuo_angle))
                Tort_mat.append(tort(you_len, you_angle))

                K_mat.append(BE)
                lambda_mat.append(asy_ratio)

        Diam_CV1.append(np.std(Diam_mat) / np.mean(Diam_mat))
        Len_CV1.append(np.std(Len_mat) / np.mean(Len_mat))
        Tort_CV1.append(np.std(Tort_mat) / np.mean(Tort_mat))
        K_CV1.append(np.std(K_mat) / np.mean(K_mat))
        lambda_CV1.append(np.std(lambda_mat) / np.mean(lambda_mat))
        angle_CV1.append(np.std(angle_mat) / np.mean(angle_mat))

    ori_cv = ['原始数据CV',  # 存Excel
              '%s±%s' % (np.around(np.mean(Diam_CV1), 3), np.around(np.std(Diam_CV1), 3)),
              '%s±%s' % (np.around(np.mean(Len_CV1), 3), np.around(np.std(Len_CV1), 3)),
              '%s±%s' % (np.around(np.mean(Tort_CV1), 3), np.around(np.std(Tort_CV1), 3)),
              '%s±%s' % (np.around(np.mean(angle_CV1), 3), np.around(np.std(angle_CV1), 3)),
              '%s±%s' % (np.around(np.mean(K_CV1), 3), np.around(np.std(K_CV1), 3)),
              '%s±%s' % (np.around(np.mean(lambda_CV1), 3), np.around(np.std(lambda_CV1), 3))
              ]
    print('完成真实数据的指标计算！')
    print('-------------------')
    return Diam_CV1, Len_CV1, Tort_CV1, K_CV1, lambda_CV1, angle_CV1, ori_cv


def get_gen_CV(Net, TreeType):
    # 生成血管树
    Diam_CV2 = []
    Len_CV2 = []
    Tort_CV2 = []

    # all_gen = []
    #
    for net in Net:
        for i in range(0, 3):
            m = np.load('../Data/%s Trees/Tree_%s_%s.npy' % (TreeType[0], net, i))
            print('加载生成树数据：', '../Data/%s Trees/Tree_%s_%s.npy' % (TreeType[0], net, i))
            Diam_CV2.append(np.std(m[:, 3]) / np.mean(m[:, 3]))
            Len_CV2.append(np.std(m[:, 4]) / np.mean(m[:, 4]))
            Tort_CV2.append(np.std(m[:, 5]) / np.mean(m[:, 5]))

    # 数据来源：D_2TreeValCV.m
    feature = io.loadmat('../Data/Temp/CV_ven Trees2.mat')
    CV_mat = feature['CV_mat'][:]

    k_CV2 = CV_mat[:, 0]
    Lambda_CV2 = CV_mat[:, 0]
    angle_CV2 = CV_mat[:, 2]

    gen_cv = ['生成数据CV',  # 沈： 存Excel
              '%s±%s' % (np.around(np.mean(Diam_CV2), 3), np.around(np.std(Diam_CV2), 3)),
              '%s±%s,' % (np.around(np.mean(Len_CV2), 3), np.around(np.std(Len_CV2), 3)),
              '%s±%s' % (np.around(np.mean(Tort_CV2), 3), np.around(np.std(Tort_CV2), 3)),
              '%s±%s' % (np.around(np.mean(CV_mat[:, 2]), 3), np.around(np.std(CV_mat[:, 2]), 3)),
              '%s±%s' % (np.around(np.mean(CV_mat[:, 0]), 3), np.around(np.std(CV_mat[:, 0]), 3)),
              '%s±%s' % (np.around(np.mean(CV_mat[:, 1]), 3), np.around(np.std(CV_mat[:, 1]), 3))
              ]
    print('完成生成数据的指标计算！')
    print('-------------------')
    return Diam_CV2, Len_CV2, Tort_CV2, k_CV2, Lambda_CV2, angle_CV2, gen_cv


if __name__ == '__main__':
    Net = ['389', '546', '913']
    # NetType,TreeType = ['Bifurcation'],['Art']
    NetType, TreeType = ['Convergence'], ['Ven']

    Diam_CV1, Len_CV1, Tort_CV1, K_CV1, Lambda_CV1, angle_CV1, ori_cv = get_ori_CV(Net, NetType)
    Diam_CV2, Len_CV2, Tort_CV2, K_CV2, Lambda_CV2, angle_CV2, gen_cv = get_gen_CV(Net, TreeType)

    p = ['P',  # 存Excel
         '%s,%s' % stats.mannwhitneyu(Diam_CV1, Diam_CV2),
         '%s,%s' % stats.mannwhitneyu(Len_CV1, Len_CV2),
         '%s,%s' % stats.mannwhitneyu(Tort_CV1, Tort_CV2),
         '%s,%s' % stats.mannwhitneyu(K_CV1, K_CV2),
         '%s,%s' % stats.mannwhitneyu(Lambda_CV1, Lambda_CV2),
         '%s,%s' % stats.mannwhitneyu(angle_CV1, angle_CV2)
         ]
    writer = pd.ExcelWriter("../Data/CV_%s.xlsx" % NetType[0])

    text2 = pd.DataFrame(ori_cv, index=[' ', 'Diam', 'Len', 'Tort', 'Angle', 'K', "lambda"])
    text2.to_excel(writer, sheet_name='CV', header=False)
    text3 = pd.DataFrame(gen_cv)
    text3.to_excel(writer, sheet_name='CV', startcol=2, header=False, index=False)
    text4 = pd.DataFrame(p)
    text4.to_excel(writer, sheet_name='CV', startcol=3, header=False, index=False)
    writer.save()
    writer.close()
    print('完成Excel表格的生成！')
    print('-------------------')
