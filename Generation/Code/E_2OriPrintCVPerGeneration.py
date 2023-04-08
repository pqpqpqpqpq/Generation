"""
输入：Excel来自 D_3OriTreetoExcel.py
输出：分层CV绘图
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import io
import pandas as pd

# 一个数据的
Net = ['389', '546', '913']
NetType,TreeType = 'Bifurcation',['Art']
# NetType, TreeType = 'Convergence', ['Ven']


for net in Net:
    data = pd.read_excel('../Support_Material/D_3OriTreetoExcel/%s/%s.xlsx' % (NetType, net))
    data = np.array(data)
    layer = data[:, 5]  # 层数列
    max_seg = data.shape[0]  # 有几行
    max_layer = np.max(layer)  # 有几层

    Diam = []
    Len = []
    Tort = []
    Angle = []
    D_CV = []
    L_CV = []
    T_CV = []
    A_CV = []

    # 第一层是不算的

    for j in range(int(max_layer - 1)):  # 遍历每个层级
        count = 0
        for i in range(int(max_seg)):  # 遍历每个分支，找出对应层级的分支

            seg = data[i]
            layer_index = seg[5]
            if layer_index == j + 2:  # 从层数为2开始
                count += 1
                Diam.append(seg[0])
                Len.append(seg[3])
                Tort.append(seg[1])
                Angle.append(seg[2])
        print('第', j + 2, '层,含', count, '个分叉')
        D_CV.append((np.std(Diam) / np.mean(Diam)))
        L_CV.append((np.std(Len) / np.mean(Len)))
        T_CV.append((np.std(Tort) / np.mean(Tort)))
        A_CV.append(np.std(Angle) / np.mean(Angle))

    # 绘制CV曲线
    plt.xlabel('Generation', size=15, family='Arial')
    plt.ylabel('CV', size=15, family='Arial')

    plt.ylim((-0.2, 1.6))
    plt.xticks(np.arange(0, len(L_CV), 2))
    plt.yticks(np.arange(-0.2, 1.6, 0.2))

    plt.plot(list(range(0, len(L_CV))), L_CV, marker="^", linewidth=2, linestyle=":", color="#ff4683")
    plt.plot(list(range(0, len(D_CV))), D_CV, marker="o", linewidth=2, linestyle="--", color="#4e9dec")
    plt.plot(list(range(0, len(T_CV))), T_CV, marker="v", linewidth=2, linestyle="-.", color="orange")
    plt.plot(list(range(0, len(A_CV))), A_CV, marker="s", linewidth=2, linestyle="-.", color="green")

    # 设置图例并且设置图例的字体及大小
    font = {'family': 'Arial',
            'weight': 'normal',
            'size': 12,
            }

    plt.grid(linestyle='--')
    plt.legend(["${CV_{Length}}$", "${CV_{Diameter}}$", "${CV_{Tortuosity}}$", "${CV_{Angle}}$"], loc="upper right",
               prop=font)
    plt.title('%s_ori_%s'% (NetType,net))
    plt.savefig("../Support_Material/%s_ori_%s_each_epoch_cv.png" % (NetType,net), dpi=600)
    plt.show()
