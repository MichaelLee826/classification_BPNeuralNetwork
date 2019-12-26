# classification_BPNeuralNetwork

> 本文介绍了利用BP神经网络实现对不同半径的圆进行多分类（3分类），特征即为圆的半径。
> 输入层12节点，一个6节点的隐藏层，输出层3个节点。

## 1.目标
通过BP算法实现对不同半径的圆的分类。
## 2.开发环境
IDE：PyCharm 2018.3.3(Community Edition)
Python及相关库的版本号如下图所示：
![版本号](https://img-blog.csdnimg.cn/20191226150201533.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21pY2hhZWxfZjIwMDg=,size_16,color_FFFFFF,t_70)
## 3.准备数据
**目的：** 生成3类圆在第一象限内的坐标（圆心都是原点）
第1类：半径范围为1~10，分类标识为‘0’
第2类：半径范围为10~20，分类标识为‘1’
第3类：半径范围为20~30，分类标识为‘2’

代码如下：`data_generate.py`
```python
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


```
共3类圆，每类生成50个圆，每个圆有12个坐标，因此在输出文件`D:\circles.csv`中总共有3×50×12=1800行数据：
![circles.csv文件](https://img-blog.csdnimg.cn/20191226154520930.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21pY2hhZWxfZjIwMDg=,size_16,color_FFFFFF,t_70)
通过生成的坐标绘制散点图如下：
![圆的散点图](https://img-blog.csdnimg.cn/2019122615475623.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21pY2hhZWxfZjIwMDg=,size_16,color_FFFFFF,t_70)
图中蓝色的点是label为0的圆，绿色的点是label为1的圆，红色的点是label为2的圆。
## 4.处理数据
**目标：** 根据第3步获得的坐标，计算每个圆的半径（勾股定理）作为神经网络的输入。

代码如下：`data_process.py`

```python
import csv
import math


def process(file_name):
    # 要读取的CSV文件
    csv_file = csv.reader(open(file_name, encoding='utf-8'))

    # 要生成的CSV文件
    out_file = open("D:\\circles_data.csv", 'a', newline='')
    csv_write = csv.writer(out_file, dialect='excel')

    # 将csv_file每一行的圆坐标取出，如果是同一批次的（同一个圆），则写入到out_file的一行中
    rows = [row for row in csv_file]
    current_batch = 'unknown'
    current_label = 'unknown'
    data_list = []
    for r in rows:
        # 将无关字符都替换为空格
        temp_string = str(r).replace('[', '').replace(']', '').replace('\'', '')
        # 将字符串以逗号分隔
        item = str(temp_string).split(',')
        # 分别取出x轴坐标、y轴坐标、批次、标签
        x = float(item[0])
        y = float(item[1])
        batch = item[2]
        label = item[3]

        # 如果是同一批次（同一个圆），则都放入data_list中
        if current_batch == batch:
            # 根据勾股定理计算半径
            distance = math.sqrt(pow(x, 2) + pow(y, 2))
            data_list.append(distance)
        # 如果不是同一批次（同一个圆），则在末尾加上标签后，作为一行写入输出文件
        else:
            if len(data_list) != 0:
                # 这个地方需注意一下，最后的标签用3列来表示，而不是一列
                if label.strip() == '0':
                    data_list.append(1)
                    data_list.append(0)
                    data_list.append(0)
                elif label.strip() == '1':
                    data_list.append(0)
                    data_list.append(1)
                    data_list.append(0)
                else:
                    data_list.append(0)
                    data_list.append(0)
                    data_list.append(1)

                result_string = str(data_list).replace('[', '').replace(']', '').replace('\'', '').strip()
                csv_write.writerow(result_string.split(','))

            # 清空data_list，继续写入下一个批次
            data_list.clear()
            distance = math.sqrt(pow(x, 2) + pow(y, 2))
            data_list.append(distance)
            current_batch = batch
            current_label = label

    # 确保最后一个批次的数据能写入
    if current_label.strip() == '0':
        data_list.append(1)
        data_list.append(0)
        data_list.append(0)
    elif current_label.strip() == '1':
        data_list.append(0)
        data_list.append(1)
        data_list.append(0)
    else:
        data_list.append(0)
        data_list.append(0)
        data_list.append(1)

    result_string = str(data_list).replace('[', '').replace(']', '').replace('\'', '').strip()
    csv_write.writerow(result_string.split(','))

    # 关闭输出文件
    out_file.close()


if __name__ == "__main__":
    process('D:\\circles.csv')

```
需要注意的是，生成的CSV文件共有15列，前12列为坐标对应的半径值，最后三列组合起来表示分类（label）：
![circles_data.csv文件](https://img-blog.csdnimg.cn/20191226164449424.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21pY2hhZWxfZjIwMDg=,size_16,color_FFFFFF,t_70)
(1,0,0)表示类型为“0”的圆，(0,1,0)表示类型为“1”的圆，(0,0,1)表示类型为“2”的圆，这样做的目的是为了下一步使用神经网络时处理起来方便。
## 5.构建BP神经网络
上一步处理好的数据可以作为训练数据，命名为：`circles_data_training.csv`
重复第3步和第4步，可以生成另一批数据作为测试数据，命名为：`circles_data_test.csv`
当然，也可以手动划分出训练数据和测试数据。
训练数据和测试数据在输入时，做了矩阵的转置，将列转置为行。

代码如下：`data_analysis_bpnn.py`

```python
import pandas as pd
import numpy as np
import datetime
from sklearn.utils import shuffle


# 1.初始化参数
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)

    # 权重和偏置矩阵
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))

    # 通过字典存储参数
    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    return parameters


# 2.前向传播
def forward_propagation(X, parameters):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    # 通过前向传播来计算a2
    z1 = np.dot(w1, X) + b1     # 这个地方需注意矩阵加法：虽然(w1*X)和b1的维度不同，但可以相加
    a1 = np.tanh(z1)            # 使用tanh作为第一层的激活函数
    z2 = np.dot(w2, a1) + b2
    a2 = 1 / (1 + np.exp(-z2))  # 使用sigmoid作为第二层的激活函数

    # 通过字典存储参数
    cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}

    return a2, cache


# 3.计算代价函数
def compute_cost(a2, Y, parameters):
    m = Y.shape[1]      # Y的列数即为总的样本数

    # 采用交叉熵（cross-entropy）作为代价函数
    logprobs = np.multiply(np.log(a2), Y) + np.multiply((1 - Y), np.log(1 - a2))
    cost = - np.sum(logprobs) / m

    return cost


# 4.反向传播（计算代价函数的导数）
def backward_propagation(parameters, cache, X, Y):
    m = Y.shape[1]

    w2 = parameters['w2']

    a1 = cache['a1']
    a2 = cache['a2']

    # 反向传播，计算dw1、db1、dw2、db2
    dz2 = a2 - Y
    dw2 = (1 / m) * np.dot(dz2, a1.T)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.multiply(np.dot(w2.T, dz2), 1 - np.power(a1, 2))
    dw1 = (1 / m) * np.dot(dz1, X.T)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

    grads = {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}

    return grads


# 5.更新参数
def update_parameters(parameters, grads, learning_rate=0.0075):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    dw1 = grads['dw1']
    db1 = grads['db1']
    dw2 = grads['dw2']
    db2 = grads['db2']

    # 更新参数
    w1 = w1 - dw1 * learning_rate
    b1 = b1 - db1 * learning_rate
    w2 = w2 - dw2 * learning_rate
    b2 = b2 - db2 * learning_rate

    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    return parameters


# 建立神经网络
def nn_model(X, Y, n_h, n_input, n_output, num_iterations=10000, print_cost=False):
    np.random.seed(3)

    n_x = n_input           # 输入层节点数
    n_y = n_output          # 输出层节点数

    # 1.初始化参数
    parameters = initialize_parameters(n_x, n_h, n_y)

    # 梯度下降循环
    for i in range(0, num_iterations):
        # 2.前向传播
        a2, cache = forward_propagation(X, parameters)
        # 3.计算代价函数
        cost = compute_cost(a2, Y, parameters)
        # 4.反向传播
        grads = backward_propagation(parameters, cache, X, Y)
        # 5.更新参数
        parameters = update_parameters(parameters, grads)

        # 每1000次迭代，输出一次代价函数
        if print_cost and i % 1000 == 0:
            print('迭代第%i次，代价函数为：%f' % (i, cost))

    return parameters


# 对模型进行测试
def predict(parameters, x_test, y_test):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    z1 = np.dot(w1, x_test) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = 1 / (1 + np.exp(-z2))

    # 结果的维度
    n_rows = y_test.shape[0]
    n_cols = y_test.shape[1]

    # 预测值结果存储
    output = np.empty(shape=(n_rows, n_cols), dtype=int)

    # 取出每条测试数据的预测结果
    for i in range(n_cols):
        # 将每条测试数据的预测结果（概率）存为一个行向量
        temp = np.zeros(shape=n_rows)
        for j in range(n_rows):
            temp[j] = a2[j][i]

        # 将每条结果（概率）从小到大排序，并获得相应下标
        sorted_dist = np.argsort(temp)
        length = len(sorted_dist)

        # 将概率最大的置为1，其它置为0
        for k in range(length):
            if k == sorted_dist[length - 1]:
                output[k][i] = 1
            else:
                output[k][i] = 0

    print('预测结果：')
    print(output)
    print('真实结果：')
    print(y_test)

    count = 0
    for k in range(0, n_cols):
        if output[0][k] == y_test[0][k] and output[1][k] == y_test[1][k] and output[2][k] == y_test[2][k]:
            count = count + 1

    acc = count / int(y_test.shape[1]) * 100
    print('准确率：%.2f%%' % acc)


if __name__ == "__main__":
    # 读取数据
    data_set = pd.read_csv('D:\\circles_data_training.csv', header=None)
    data_set = shuffle(data_set)            # 打乱数据的输入顺序
    # 取出“特征”和“标签”，并做了转置，将列转置为行
    X = data_set.ix[:, 0:11].values.T       # 前12列是特征
    Y = data_set.ix[:, 12:14].values.T      # 后3列是标签
    Y = Y.astype('uint8')

    # 开始训练
    start_time = datetime.datetime.now()
    # 输入12个节点，隐层6个节点，输出3个节点，迭代10000次
    parameters = nn_model(X, Y, n_h=6, n_input=12, n_output=3, num_iterations=10000, print_cost=True)
    end_time = datetime.datetime.now()
    print("用时：" + str((end_time - start_time).seconds) + 's' + str(round((end_time - start_time).microseconds / 1000)) + 'ms')

    # 对模型进行测试
    data_test = pd.read_csv('D:\\circles_data_test.csv', header=None)
    x_test = data_test.ix[:, 0:11].values.T
    y_test = data_test.ix[:, 12:14].values.T
    y_test = y_test.astype('uint8')
    predict(parameters, x_test, y_test)


```
上述代码可以参考这篇文章：[纯Python实现鸢尾属植物数据集神经网络模型](https://yq.aliyun.com/articles/614411?utm_content=m_1000007130)。

代码中需要注意的几个**关键参数**：

 1. learning_rate=0.0075，学习率（可调）
 2. n_h=6，隐藏层节点数（可调）
 3. n_input=12，输入层节点数
 4. n_output=3，输出层节点数
 5. num_iterations=10000，迭代次数（可调）

另外，对于`predict(parameters, x_test, y_test)`函数需要说明一下：
`a2`矩阵是最终的预测结果，但是是以概率的形式表示的（可以打印看一下）。通过比较3个类的概率，选出概率最大的那个置为1，其它两个置为0，形成`output`矩阵。

运行结果：
![运行结果](https://img-blog.csdnimg.cn/20191226172305537.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21pY2hhZWxfZjIwMDg=,size_16,color_FFFFFF,t_70)
上图中第一红框表示神经网络预测出的分类结果，第二个红框表示测试集中真实的分类（(1,0,0)表示这个圆属于类型“0”）。

每次运行时，正确率可能不一样，最高能达到100%。通过调整刚才提到的**关键参数**中的**学习率、隐藏层节点数、迭代次数**可以提高正确率。

## 总结
&emsp;&emsp;神经网络的输入为12个半径值，输出结果为一个3维向量，其中置1的位就是对应的分类。
&emsp;&emsp;在实际应用中，12个半径值对应12个特征，3维向量表示能分3类。只要根据实际应用的需要修改特征数和分类数即可将上述程序应用于不同分类场景。
#### 以上就是利用BP神经网络实现多特征多分类的全部过程。

