import os
import shutil
from sklearn.model_selection import train_test_split

# 定义数据集路径
dataset_dir = "dataset"  # 数据集根目录
output_dir = "split_dataset"  # 划分后的数据集保存路径

# 定义划分比例
train_ratio = 0.7  # 训练集比例
val_ratio = 0.15   # 验证集比例
test_ratio = 0.15  # 测试集比例

# 创建输出目录
os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

# 遍历每个类别
for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_dir):
        continue

    # 获取当前类别的所有文件
    files = [os.path.join(class_dir, f) for f in os.listdir(
        class_dir) if os.path.isfile(os.path.join(class_dir, f))]

    # 划分数据集
    train_files, test_files = train_test_split(
        files, test_size=test_ratio, random_state=42)
    train_files, val_files = train_test_split(
        train_files, test_size=val_ratio / (1 - test_ratio), random_state=42)

    # 将文件复制到对应的目录
    def copy_files(files, split_name):
        split_dir = os.path.join(output_dir, split_name, class_name)
        os.makedirs(split_dir, exist_ok=True)
        for file in files:
            shutil.copy(file, split_dir)

    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")

print("数据集划分完成！")
