import os
import csv

import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split

# 定义文件夹路径
'''base_dir = '/lab/kirito/data/GenImage/wukong/imagenet_ai_0424_wukong'
train_dir = os.path.join(base_dir, 'val')
ai_dir = os.path.join(train_dir, 'ai')
nature_dir = os.path.join(train_dir, 'nature')

# 获取ai和nature文件夹下的所有图片文件
ai_images = [os.path.join(ai_dir, img) for img in os.listdir(ai_dir) if img.lower().endswith(('png', 'jpg', 'jpeg'))]
nature_images = [os.path.join(nature_dir, img) for img in os.listdir(nature_dir) if
                 img.lower().endswith(('png', 'jpg', 'jpeg'))]

# 从ai和nature文件夹中分别选取2500的图片作为测试集
ai_train, ai_test = train_test_split(ai_images, test_size=0.05, random_state=42)
nature_train, nature_test = train_test_split(nature_images, test_size=0.05, random_state=42)
ai_test = random.sample(ai_images, 3500)
nature_test = random.sample(nature_images, 3500)

# 合并ai和nature的测试集
test_images = ai_images + nature_images
labels = [1] * len(ai_images) + [0] * len(nature_images)

# 创建一个DataFrame，并保存为test.csv
test_df = pd.DataFrame({
    'img_path': test_images,
    'label': labels
})

# 保存为CSV文件
test_csv_path = os.path.join(base_dir, 'test.csv')
test_df.to_csv(test_csv_path, index=False)

print(f'Test set created and saved to {test_csv_path}')'''


base_dir = '/lab/kirito/data/CNNspot_test/test/ldm'
real_dir = os.path.join(base_dir, '0_real')
fake_dir = os.path.join(base_dir, '1_fake')
csv_filename = os.path.join(base_dir, 'test.csv')

data = []

for folder, label in [(real_dir, '0'), (fake_dir, '1')]:
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.abspath(os.path.join(folder, filename))
            data.append([image_path, label])

with open(csv_filename, 'w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_path', 'label'])
    writer.writerows(data)


'''base_dir = '/lab/kirito/data/CNNspot_test/test/gansformer'
csv_filename = os.path.join(base_dir, 'test.csv')

data = []

for category in os.listdir(base_dir):
    category_path = os.path.join(base_dir, category)
    if os.path.isdir(category_path):
        for subfolder,label in [('1_fake',1)]:
            subfolder_path = os.path.join(category_path, subfolder)
            if os.path.exists(subfolder_path):
                for filename in os.listdir(subfolder_path):
                    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        image_path = os.path.abspath(os.path.join(subfolder_path, filename))
                        data.append([image_path, label])

with open(csv_filename, 'w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_path', 'label'])
    writer.writerows(data)'''


'''import os
import csv

# 定义文件夹路径
progan_dir = '/lab/kirito/data/CNNspot_test/test/projectedgan'

# 获取progan文件夹下的所有种类文件夹
categories = [d for d in os.listdir(progan_dir) if os.path.isdir(os.path.join(progan_dir, d))]

for category in categories:
    # 定义0_real和1_fake文件夹路径
    real_dir = os.path.join(progan_dir, category, '0_real')
    fake_dir = os.path.join(progan_dir, category, '1_fake')
    
    # 定义csv文件路径
    csv_path = os.path.join(progan_dir, f'{category}_test.csv')
    
    # 打开csv文件，写入数据
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(['img_path', 'label'])
        
        # 遍历0_real文件夹，写入图片路径和标签
        if os.path.exists(real_dir):
            for img_name in os.listdir(real_dir):
                img_path = os.path.abspath(os.path.join(real_dir, img_name))
                writer.writerow([img_path, 0])
        
        # 遍历1_fake文件夹，写入图片路径和标签
        if os.path.exists(fake_dir):
            for img_name in os.listdir(fake_dir):
                img_path = os.path.abspath(os.path.join(fake_dir, img_name))
                writer.writerow([img_path, 1])

print("CSV文件生成完毕。")'''


'''import os
import pandas as pd
from sklearn.utils import shuffle

# 定义包含CSV文件的文件夹路径
csv_folder = '/lab/kirito/data/CNNspot/test/stylegan2'

# 获取文件夹中的所有CSV文件
csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

# 初始化一个空的DataFrame来存储所有数据
all_data = pd.DataFrame()

# 遍历CSV文件，读取并合并数据
for csv_file in csv_files:
    file_path = os.path.join(csv_folder, csv_file)
    data = pd.read_csv(file_path)
    all_data = pd.concat([all_data, data], ignore_index=True)

# 打乱数据
shuffled_data = shuffle(all_data)

# 保存打乱后的数据到新的CSV文件
for csv_file in csv_files:
    # 构建新的文件名
    new_file_name = csv_file.replace('.csv', '_shuffled.csv')
    new_file_path = os.path.join(csv_folder, new_file_name)
    
    # 保存打乱后的数据
    shuffled_data.to_csv(new_file_path, index=False)

print("打乱后的数据已保存到新的CSV文件中。")'''

