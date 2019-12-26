import numpy as np
import math
import random
import csv


# 只生成第一象限内的坐标即可。每个圆生成12个坐标(x,y)，相当于12个特征维度
def generate_circle(lower, upper):
    # 圆在第一象限内的坐标
    data_ur = np.zeros(shape=(12, 2))

    # 在上下限范围内，随机产生一个值作为半径
    radius = random.randint(int(lower), int(upper))

    # 在0~90度内，每隔7.5度取一次坐标，正好取12次
    angles = np.arange(0, 0.5 * np.pi, 1 / 24 * np.pi)
    for i in range(12):
        temp_ur = np.zeros(2)
        x = round(radius * math.cos(angles[i]), 2)
        y = round(radius * math.sin(angles[i]), 2)
        temp_ur[0] = x
        temp_ur[1] = y
        data_ur[i] = temp_ur

    return data_ur, label


# 将坐标保存到CSV文件中
def save2csv(data, batch, label):
    out = open("D:\\circles.csv", 'a', newline='')
    csv_write = csv.writer(out, dialect='excel')

    length = int(data.size / 2)
    for i in range(length):
        string = str(data[i][0]) + ',' + str(data[i][1]) + ',' + str(batch) + ',' + str(label)
        temp = string.split(',')
        csv_write.writerow(temp)
    out.close()


if __name__ == "__main__":
    '''
        生成3类圆，标签（label）分别为：0、1、2
        第1类圆的半径下限为1，上限为10
        第2类圆的半径下限为10，上限为20
        第3类圆的半径下限为20，上限为30
        圆心都为原点
    '''
lower = [1, 10, 20]  # 半径随机值的下限
upper = [10, 20, 30]  # 半径随机值的上限
label = ['0', '1', '2']  # 种类的标签

for i in range(len(label)):
    # 每类数据生成50组
    for j in range(50):
        data, label = generate_circle(lower[i], upper[i])
        batch = 50 * i + j + 1  # 数据的批次，用来区分每个坐标是属于哪个圆的
        save2csv(data, batch, label[i])

