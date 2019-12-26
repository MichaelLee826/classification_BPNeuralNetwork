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
