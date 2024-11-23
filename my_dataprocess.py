import os
import csv
import pandas as pd

'''# 设置progan文件夹路径
progan_folder = '/lab/kirito/data/CNNspot_test/train/progan'  # 请替换为你的progan文件夹的实际路径
output_fold = '/lab/kirito/data/CNNspot_test/train/progan/datacsv'

# 获取所有类别文件夹
categories = [d for d in os.listdir(progan_folder) if os.path.isdir(os.path.join(progan_folder, d))]

# 遍历每个类别文件夹
for category in categories:
    category_path = os.path.join(progan_folder, category)

    # 遍历0_real和1_fake文件夹
    for sub_folder in ['0_real', '1_fake']:
        sub_folder_path = os.path.join(category_path, sub_folder)
        if not os.path.exists(sub_folder_path):
            continue

        # 获取所有图片的路径
        img_paths = [os.path.join(sub_folder_path, img) for img in os.listdir(sub_folder_path) if
                     os.path.isfile(os.path.join(sub_folder_path, img))]

        # 确定CSV文件名和标签
        csv_file_name = f"{category}_{sub_folder.split('_')[1]}.csv"
        label = 0 if sub_folder == '0_real' else 1

        # 写入CSV文件
        csv_file_path = os.path.join(output_fold, csv_file_name)
        with open(csv_file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['img_path', 'label'])
            for img_path in img_paths:
                writer.writerow([img_path, label])

print("CSV文件生成完毕。")'''

'''def counr_csv_rows(csv_folder):
    files = os.listdir(csv_folder)

    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(csv_folder, file)
            try:
                df = pd.read_csv(file_path)
                row_count = len(df)
                print(f"file:{file}, row_count:{row_count}")
            except Exception as e:
                print(e)


csv_folser = '/lab/kirito/data/CNNspot/train/progan/datacsv'
counr_csv_rows(csv_folser)'''

import os
import csv

def generate_csv_from_images(folder_path, label):
    # 获取文件夹的名称
    folder_name = os.path.basename(folder_path)
    # 定义CSV文件名
    csv_filename = f"{folder_name}.csv"
    
    # 获取所有图片文件的路径
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', 'JPEG'))]
    
    # 写入CSV文件
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['img_path', 'label'])  # 写入表头
        for img_path in image_paths:
            writer.writerow([os.path.abspath(img_path), label])  # 写入图片路径和标签
    
    print(f"CSV file '{csv_filename}' created successfully!")

# 使用示例
folder_path = "/lab/kirito/data/cliptrain/train/1_real_noisy_png"  # 替换为你的文件夹路径
label = 0  # 替换为你设定的标签
generate_csv_from_images(folder_path, label)
