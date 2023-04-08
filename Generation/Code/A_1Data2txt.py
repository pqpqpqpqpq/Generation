"""
将原始数据opt，dat文件变为txt文件方便后续处理
"""

netlist = ['Men_546', 'Men_389', 'Men_913']

for i in range(len(netlist)):
    net = netlist[i]

    # opt to txt
    with open('../Data/Source Data/%s.opt' % net, 'r') as f:
        file = f.read()
        file = file.strip().split('\n')
        for data in file:
            data = data.replace(',', ' ')
            data = data.replace('segname', '-1')
            data = data.replace("\"", " ")
            data = data.replace(" ", "  ")
            with open('../Data/Source Data/%s_opt.txt' % net, 'a') as f:
                f.write(data + '\n')

    # dat to txt
    with open('../Data/Source Data/%s.dat' % net, 'r') as f:
        file = f.read()
        file = file.strip().split('\n')
        for data in file:
            data = data.replace(',', ' ')
            data = data.replace('segname', '-1')
            data = data.replace("\"", " ")
            data = data.replace(" ", "  ")
            with open('../Data/Source Data/%s_dat.txt' % net, 'a') as f:
                f.write(data + '\n')

print('完成转换：文件存入','../Data/Source Data')