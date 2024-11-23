import os
from tqdm import tqdm
import pandas as pd

'''image_root = '/home/dell/Documents/Appledog/data/GenImage'

count = 0
for r, d, f in tqdm(os.walk(image_root)):
    if len(f) != 0:
        for sub_f in f:
            if sub_f.endswith('.jpg') or sub_f.endswith('.png') or sub_f.endswith('.jpeg') or sub_f.endswith('.JPG') or sub_f.endswith('.JPEG') or sub_f.endswith('.PNG'):
                image_path = os.path.join(r, sub_f)
                image_size = os.path.getsize(image_path)
                if image_size < 10:
                    os.remove(image_path)
                    count += 1
print('Deleted ' + str(count) + ' images')
print('finished!')'''

# 读取CSV文件
file_path = '/lab/kirito/data/CNNspot_test/test/ldm/test.csv'
df = pd.read_csv(file_path)

# 打乱数据
shuffled_df = df.sample(frac=1).reset_index(drop=True)

# 保存打乱后的数据到CSV文件
shuffled_file_path = '/lab/kirito/data/CNNspot_test/test/ldm//test_1.csv'
shuffled_df.to_csv(shuffled_file_path, index=False)

print(f'Data has been shuffled and saved to {shuffled_file_path}')

'''import os
import random
import shutil

# 原始和目标目录
source_dir = '/lab/kirito/data/CNNspot/train/progan'
target_dir = '/lab/kirito/data/CNNspot_test/train/progan'

# 每个类别下选择的图片数量
num_images = 300

# 创建目标目录
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 获取类别列表
categories = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

for category in categories:
    real_dir = os.path.join(source_dir, category, '0_real')
    fake_dir = os.path.join(source_dir, category, '1_fake')
    
    # 检查0_real和1_fake文件夹是否存在
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        print(f"Missing folders in category {category}")
        continue
    
    # 创建目标目录的类别结构
    target_real_dir = os.path.join(target_dir, category, '0_real')
    target_fake_dir = os.path.join(target_dir, category, '1_fake')
    os.makedirs(target_real_dir, exist_ok=True)
    os.makedirs(target_fake_dir, exist_ok=True)

    # 随机选择图片并复制
    real_images = os.listdir(real_dir)
    fake_images = os.listdir(fake_dir)
    
    if len(real_images) < num_images or len(fake_images) < num_images:
        print(f"Not enough images in category {category}")
        continue
    
    selected_real_images = random.sample(real_images, num_images)
    selected_fake_images = random.sample(fake_images, num_images)
    
    for image in selected_real_images:
        shutil.copy(os.path.join(real_dir, image), os.path.join(target_real_dir, image))
        
    for image in selected_fake_images:
        shutil.copy(os.path.join(fake_dir, image), os.path.join(target_fake_dir, image))

print("Image selection and copying complete.")'''

