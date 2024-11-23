import os
import csv

# 原始图片目录
source_dir = '/lab/kirito/data/CNNspot_test/val'
output_file = '/lab/kirito/data/CNNspot_test/val/val.csv'

# 获取类别列表
categories = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

# 打开CSV文件写入数据
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['img_path', 'label'])

    for category in categories:
        real_dir = os.path.join(source_dir, category, '0_real')
        fake_dir = os.path.join(source_dir, category, '1_fake')
        
        # 检查0_real和1_fake文件夹是否存在
        if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
            print(f"Missing folders in category {category}")
            continue
        
        # 记录真实图片
        real_images = os.listdir(real_dir)
        for image in real_images:
            img_path = os.path.abspath(os.path.join(real_dir, image))
            writer.writerow([img_path, 0])
        
        # 记录假图片
        fake_images = os.listdir(fake_dir)
        for image in fake_images:
            img_path = os.path.abspath(os.path.join(fake_dir, image))
            writer.writerow([img_path, 1])

print(f"CSV file '{output_file}' created successfully.")
