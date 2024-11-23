import os
import csv

# 定义文件夹路径
root_folder = '/lab/kirito/data/CNNspot_test/val'
categories = os.listdir(root_folder)
# categories = ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair']

# 定义输出CSV文件路径
realtrain_csv = '/lab/kirito/data/CNNspot_test/train/realval.csv'
faketrain_csv = '/lab/kirito/data/CNNspot_test/train/fakeval.csv'

# 打开两个CSV文件用于写入
with open(realtrain_csv, 'w', newline='') as real_file, open(faketrain_csv, 'w', newline='') as fake_file:
    real_writer = csv.writer(real_file)
    fake_writer = csv.writer(fake_file)

    # 写入CSV文件的表头
    real_writer.writerow(['img_path', 'label'])
    fake_writer.writerow(['img_path', 'label'])

    # 遍历每个类别文件夹
    for category in categories:
        category_path = os.path.join(root_folder, category)
        if os.path.isdir(category_path):
            real_folder = os.path.join(category_path, '0_real')
            fake_folder = os.path.join(category_path, '1_fake')

            # 处理real文件夹
            if os.path.exists(real_folder):
                for img_name in os.listdir(real_folder):
                    img_path = os.path.abspath(os.path.join(real_folder, img_name))
                    real_writer.writerow([img_path, 0])

            # 处理fake文件夹
            if os.path.exists(fake_folder):
                for img_name in os.listdir(fake_folder):
                    img_path = os.path.abspath(os.path.join(fake_folder, img_name))
                    fake_writer.writerow([img_path, 1])

print(f'{realtrain_csv} and {faketrain_csv} have been created.')
