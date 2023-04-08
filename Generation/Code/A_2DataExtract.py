"""
1-提取包涵下一级的原始分叉数
2-这里统计了原始数据包含的分支信息（动脉，静脉，毛细）
3-这里统计了原始数据包含的分叉信息（分叉总量，包涵下一级的分叉）
4-三个分叉均从交点出发
"""

import numpy as np
import operator
from C_Func import reverseList
import os

selectBranch = True

netType = 'Convergence'  # Bifurcation, Convergence
net = 'Men_389'  # Men_546/Men_913/Men_389/
data_opt = np.loadtxt('../Data/Source Data/%s_opt.txt' % net)
data_dat = np.loadtxt('../Data/Source Data/%s_dat.txt' % net)

print('处理数据：%s_%s' % (net,netType))

x1 = data_opt[:, 5]  # x起点坐标
x2 = data_opt[:, 9]  # x终点坐标
y1 = data_opt[:, 6]  # y起点坐标
y2 = data_opt[:, 10]  # y终点坐标
z1 = data_opt[:, 4]  # Segment的起始ID
z2 = data_opt[:, 8]  # Segment的终止ID

# Men_389 opt文件没有管径信息,所有这里用第11维暂时代替管径,做处理,后面再替换为管径信息
if net == 'Men_389':
    diam = data_opt[:, 11]
else:
    diam = data_opt[:, 12]

# Men_913也比较特殊,这里把他那个大于4000的都变为-1
if net == 'Men_913':
    for i in range(len(z1)):
        if z1[i] > 4000:
            z1[i] = -1
    for i in range(len(z2)):
        if z2[i] > 4000:
            z2[i] = -1

# 获取每个血管段的头节点在数组中的位置编号，则[Head_ID[i]:Head_ID[i+1]]为第I段血管段的子段编号
Head_ID = [i for i in range(1, len(x1) - 1) if z1[i] == z2[i - 1] != -1 or z1[i] != z2[i - 1]]
Head_ID = [0] + Head_ID + [len(data_opt)]

# 将同一个segment的头尾结点xy坐标管径放到一起
Segment = []
for i in range(len(Head_ID) - 1):
    x1_part = x1[Head_ID[i]:Head_ID[i + 1]]
    x1_part = x1_part.tolist()

    y1_part = y1[Head_ID[i]:Head_ID[i + 1]]
    y1_part = y1_part.tolist()

    Seg_From = z1[Head_ID[i]]  # 血管段的头节点
    Seg_To = z2[Head_ID[i + 1] - 1]  # 血管段的尾节点
    Seg_x = x1_part + [x2[Head_ID[i + 1] - 1]]  # 血管段各个子段端点的x坐标
    Seg_y = y1_part + [y2[Head_ID[i + 1] - 1]]  # 血管段各个子段端点的x坐标

    diameter = diam[Head_ID[i + 1] - 1]

    Segment.append([Seg_From] + Seg_x + Seg_y + [diameter] + [Seg_To])  # 将头节点与x坐标拼接起来

# dat文件与opt文件的from和to有可能相反，这里把它反过来，坐标也要进行相应的反转（以dat文件的为准）
segment_mod = []
for i in range(len(Segment)):
    if data_dat[i][2] == Segment[i][-1] and data_dat[i][3] == Segment[i][0]:
        part_id_tou = [Segment[i][-1]]
        part_id_wei = [Segment[i][-2], Segment[i][0]]

        part_x = Segment[i][1:len(Segment[i]) // 2]
        part_x.reverse()
        part_y = Segment[i][len(Segment[i]) // 2:len(Segment[i]) - 2]
        part_y.reverse()

        segment_mod.append(part_id_tou + part_x + part_y + part_id_wei)
    else:
        segment_mod.append(Segment[i])
Segment = segment_mod

#  vessel type: 1 arterioles, 2 capillary, 3 venules
branch = []  # branch:[Ves_Type, Seg_From, Seg_x1, ..., Seg_xn, Seg_y1, ..., Seg_yn, diameter, Seg_To]
for i in range(len(Segment)):
    if data_dat[i][2] == Segment[i][0] and data_dat[i][3] == Segment[i][-1]:
        if net == 'Men_389':
            Segment[i][-2] = data_dat[i][4]  # 补充389网络之前预置的管径
            branch.append([data_dat[i][1]] + Segment[i])
        else:
            branch.append([data_dat[i][1]] + Segment[i])

# 区分动静脉毛细血管
art = []
cap = []
ven = []

for i in range(len(branch)):
    if branch[i][0] == 1 or branch[i][0] == 0:
        art.append(branch[i])
    if branch[i][0] == 2:
        cap.append(branch[i])
    if branch[i][0] == 3 or branch[i][0] == 0:
        ven.append(branch[i])

print('动脉', len(art))
print('静脉', len(ven))
print('毛细', len(cap))

# bifurcation:[Seg_x1, ..., Seg_xn, Seg_y1, ..., Seg_yn, diameter]
bifurcation = []
branch.sort(key=operator.itemgetter(1))
for i in range(len(art)):
    for j in range(len(branch) - 1):
        if art[i][-1] == branch[j][1] == branch[j + 1][1]:
            bifurcation.append([art[i][2:-1]] + [branch[j][2:-1]] + [branch[j + 1][2:-1]])

# convergence:[Seg_x1, ..., Seg_xn, Seg_y1, ..., Seg_yn, diameter]
branch.sort(key=operator.itemgetter(-1))
convergence = []
for i in range(len(ven)):
    for j in range(len(branch) - 1):
        if ven[i][1] == branch[j][-1] == branch[j + 1][-1]:
            convergence.append([branch[j][2:-1]] + [branch[j + 1][2:-1]] + [ven[i][2:-1]])

# 将尾等分叉排列方式改为，与头等分叉相同的排列方式
convergence_mod = []
for i in range(len(convergence)):
    zhu = convergence[i][0]
    zuo = convergence[i][1]
    you = convergence[i][2]

    zhu_x = zhu[0:len(zhu) // 2]
    zuo_x = zuo[0:len(zuo) // 2]
    you_x = you[0:len(you) // 2]
    zhu_y = zhu[len(zhu) // 2:(len(zhu) - 1)]
    zuo_y = zuo[len(zuo) // 2:(len(zuo) - 1)]
    you_y = you[len(you) // 2:(len(you) - 1)]

    zhu_new = you_x + you_y + [you[-1]]
    zuo_new = zhu_x[::-1] + zhu_y[::-1] + [zhu[-1]]
    you_new = zuo_x[::-1] + zuo_y[::-1] + [zuo[-1]]
    convergence_mod.append([zhu_new] + [zuo_new] + [you_new])

bifurcation_mod = []
for i in range(len(bifurcation)):
    zhu = bifurcation[i][0]
    zuo = bifurcation[i][1]
    you = bifurcation[i][2]

    zhu_x = zhu[0:len(zhu) // 2]
    zuo_x = zuo[0:len(zuo) // 2]
    you_x = you[0:len(you) // 2]
    zhu_y = zhu[len(zhu) // 2:(len(zhu) - 1)]
    zuo_y = zuo[len(zuo) // 2:(len(zuo) - 1)]
    you_y = you[len(you) // 2:(len(you) - 1)]

    zhu_new = zhu_x[::-1] + zhu_y[::-1] + [zhu[-1]]
    zuo_new = zuo_x + zuo_y + [zuo[-1]]
    you_new = you_x + you_y + [you[-1]]
    bifurcation_mod.append([zhu_new] + [zuo_new] + [you_new])

# 若想把分叉和汇聚分开，则单独提取即可
if netType == 'Bifurcation':
    final = bifurcation_mod
if netType == 'Convergence':
    final = convergence_mod
if netType == 'All':
    final = convergence_mod + bifurcation_mod

# 去除重复的点
NoRepeat = []
for i in range(len(final)):
    zhu = final[i][0]
    zuo = final[i][1]
    you = final[i][2]

    zhu_diam = [zhu[-1]]
    zuo_diam = [zuo[-1]]
    you_diam = [you[-1]]

    zhu_x = zhu[0:len(zhu) // 2]
    zuo_x = zuo[0:len(zuo) // 2]
    you_x = you[0:len(you) // 2]
    zhu_y = zhu[len(zhu) // 2:(len(zhu) - 1)]
    zuo_y = zuo[len(zuo) // 2:(len(zuo) - 1)]
    you_y = you[len(you) // 2:(len(you) - 1)]

    zhu_x1 = []
    zhu_y1 = []
    for j in range(len(zhu_x) - 1):
        if zhu_x[j] != zhu_x[j + 1] or zhu_y[j] != zhu_y[j + 1]:
            zhu_x1.append(zhu_x[j])
            zhu_y1.append(zhu_y[j])
    zhu_x1.append(zhu_x[-1])
    zhu_y1.append(zhu_y[-1])

    zuo_x1 = []
    zuo_y1 = []
    for j in range(len(zuo_x) - 1):
        if zuo_x[j] != zuo_x[j + 1] or zuo_y[j] != zuo_y[j + 1]:
            zuo_x1.append(zuo_x[j])
            zuo_y1.append(zuo_y[j])
    zuo_x1.append(zuo_x[-1])
    zuo_y1.append(zuo_y[-1])

    you_x1 = []
    you_y1 = []
    for j in range(len(you_x) - 1):
        if you_x[j] != you_x[j + 1] or you_y[j] != you_y[j + 1]:
            you_x1.append(you_x[j])
            you_y1.append(you_y[j])
    you_x1.append(you_x[-1])
    you_y1.append(you_y[-1])

    zhu = zhu_x1 + zhu_y1 + zhu_diam
    zuo = zuo_x1 + zuo_y1 + zuo_diam
    you = you_x1 + you_y1 + you_diam
    NoRepeat.append([zhu] + [zuo] + [you])
final = NoRepeat

# 挑选出包含子分支的分叉
if selectBranch:
    print("挑选匹配子分支之前分叉数:", len(final))
    have_branch = []
    for i in range(len(final)):
        for j in range(len(final)):
            if final[i][1] == reverseList(final[j][0]) or final[i][2] == reverseList(final[j][0]):
                if final[i] not in have_branch:
                    have_branch.append(final[i])
    print("挑选匹配子分支之后分叉数:", len(have_branch))
    final = have_branch

# 挑选小于Node_num段的分支
Node_num = 64
select_final = []
for i in range(len(final)):
    if len(final[i][0]) <= Node_num and len(final[i][1]) <= Node_num and len(final[i][2]) <= Node_num:
        select_final.append(final[i])
print("所有分叉：", len(final), "\n点数小于%d分叉 : " % Node_num, len(select_final))
final = np.array(select_final)

if not os.path.exists('../Data/Normalized Data/'):
    os.makedirs('../Data/Normalized Data/')

# 保存数据坐标
np.save("../Data/Normalized Data/Coord&Diam_%s_%s.npy" % (net, netType), final)
