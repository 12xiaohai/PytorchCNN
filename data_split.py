import os
import shutil
import random

# 原始数据集路径
source_dir = "COVID-19_Radiography_Dataset"

# 目标数据集路径
target_dir = "dataset/COVID_19_Radiography_Dataset"

# 设置划分比例
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# 确保目标目录存在
os.makedirs(target_dir, exist_ok=True)
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(target_dir, split), exist_ok=True)

# 需要划分的类别
categories = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

for category in categories:
    # **所有类别都在各自的 images/ 目录下**
    category_path = os.path.join(source_dir, category, "images")

    # **检查目录是否存在**
    if not os.path.exists(category_path):
        print(f"⚠️ 警告: {category_path} 目录不存在，跳过...")
        continue

    print(f"📂 处理类别: {category} (路径: {category_path})")

    # 获取所有图片
    all_files = [
        f for f in os.listdir(category_path) if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    random.shuffle(all_files)  # 打乱顺序

    print(f"🔍 找到 {len(all_files)} 张图片")

    if len(all_files) == 0:
        continue

    # 计算划分数量
    total = len(all_files)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    train_files = all_files[:train_count]
    val_files = all_files[train_count: train_count + val_count]
    test_files = all_files[train_count + val_count:]

    # 复制文件到 train/val/test 目录
    for split, files in zip(
        ["train", "val", "test"], [train_files, val_files, test_files]
    ):
        category_target = os.path.join(target_dir, split, category)
        os.makedirs(category_target, exist_ok=True)

        for file in files:
            src = os.path.join(category_path, file)
            dst = os.path.join(category_target, file)
            shutil.copy2(src, dst)

    print(
        f"✅ {category}: 训练集 {train_count} | 验证集 {val_count} | 测试集 {len(test_files)}"
    )

print("🎉 数据集划分完成！")
