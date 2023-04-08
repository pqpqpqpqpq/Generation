"""
沈：
生成真实树，
动脉为红色，静脉为蓝色
像素尺寸还原为物理尺寸

输入：2_OriGetTreePerGeneration的数据，npy数据
输出：画图
"""

import numpy as np
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":
    net='913' # 389 546 913
    nettype='Convergence'
    # nettype = 'Bifurcation'
    # 加载数据


    tree_data = np.load("../Support_Material/D_3OriTreetoExcel/%s/%s.npy" % (nettype, net), allow_pickle=True)
    print("../Support_Material/D_3OriTreetoExcel/%s/%s.npy" % (nettype, net))


    x_list = []
    y_list = []
    for i in range(len(tree_data)):
        bifur = tree_data[i]
        Coord_x = bifur[0:((len(bifur) - 5) // 2)]
        Coord_y = bifur[((len(bifur) - 5) // 2):(len(bifur) - 6)]
        for j in Coord_x:
            x_list.append(j)
        for j in Coord_y:
            y_list.append(j)


    dpi = 600
    umpp = 1
    fig = plt.figure(figsize=((max(x_list) - min(x_list)) / dpi, (max(y_list) - min(y_list)) / dpi))
    ax = fig.add_axes([0, 0, 1, 1])
    for i in range(len(tree_data)):
        bifur = tree_data[i]
        bifur_diam = [bifur[-6]]
        bifur_layer = bifur[-1]
        bifur_x = bifur[0:((len(bifur) - 5) // 2)]
        bifur_y = bifur[((len(bifur) - 5) // 2):(len(bifur) - 6)]

        plt.xlim(min(x_list), max(x_list))
        plt.ylim(min(y_list), max(y_list))

        # 这样计算的像素尺寸和物理尺寸是等效的:linewidth=72 / (dpi * umpp) * bifur_diam[0]
        plt.plot(bifur_x, bifur_y, 'k-',linewidth=72 / (dpi * umpp) * bifur_diam[0])


    #  设置x轴和y轴等比例
    ax = plt.gca()
    ax.set_aspect(1)
    plt.axis('off')
    if not os.path.exists('../Data/ori trees/' ):
        os.makedirs('../Data/ori trees/')
    plt.savefig('../Data/ori trees/ori_%s_%s.tiff' % (net, nettype), dpi=dpi, pil_kwargs={"compression": "tiff_lzw"})
    plt.show(dpi=600)
    plt.close()


