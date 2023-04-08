"""
沈：计算生成树的分层的异质性并画图
没有角度的指标
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import io

# 一个数据的
net = 'Tree_389_0'
# net = 'Tree_546_0'
# net = 'Tree_913_0'
data = np.load('../Data/ven Trees/%s.npy' % net)

# data = np.load('../Data/art Trees/%s.npy' % net)
layer = data[:, 9]  # 层数列
max_seg = data.shape[0]  # 有几行
max_layer = np.max(layer)  # 有几层

Diam = []
Len = []
Tort = []
angle = []

D_CV = []
L_CV = []
T_CV = []
A_CV = []

# 第一层是不算的

for j in range(int(max_layer - 1)):  # 遍历每个层级
    count = 0
    for i in range(int(max_seg)):  # 遍历每个分支，找出对应层级的分支

        seg = data[i]
        layer_index = seg[9]
        if layer_index == j + 2:  # 从层数为2开始
            count += 1
            Diam.append(seg[3])
            Len.append(seg[4])
            Tort.append(seg[5])
            angle.append(seg[8])
    print('第', j + 2, '层,含', count, '个分叉')
    D_CV.append((np.std(Diam) / np.mean(Diam)))
    L_CV.append((np.std(Len) / np.mean(Len)))
    T_CV.append((np.std(Tort) / np.mean(Tort)))
    A_CV.append((np.std(angle) / np.mean(angle)))

# 绘制CV曲线
plt.xlabel('Generation', size=15, family='Arial')
plt.ylabel('CV', size=15, family='Arial')

plt.title(net)
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
plt.savefig("../Support_Material/%s_each_epoch_cv.png" % net, dpi=600)
plt.show()
plt.close()
