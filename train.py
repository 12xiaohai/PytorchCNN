import torch  # 导入torch库
import torch.nn as nn
import torch.optim as optim  # 优化器
from torch.utils.data import DataLoader  # 数据加载
from torchvision import datasets, transforms  # 数据集和数据变换
from tqdm import tqdm  # 训练进度条
import os
from model.cnn import CNN

device = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu")  # 设备选择 cpu或者gpu
print(device)  # 打印设备
# 对图像做变换
train_transformer = transforms.Compose([
    transforms.Resize([224, 224]),  # 将数据裁剪为224*224大小
    transforms.ToTensor(),  # 把图片转换为 tensor张量 0-1的像素值
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # 归一化 -1-1的像素值
])

test_transformer = transforms.Compose([
    transforms.Resize([224, 224]),  # 将数据裁剪为224*224大小
    transforms.ToTensor(),  # 把图片转换为 tensor张量 0-1的像素值
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # 归一化 -1-1的像素值
])

# 定义数据集加载类
# datasets.ImageFolder() 自动根据文件夹结构为图像分配标签
trainset = datasets.ImageFolder(root="./dataset/", transform=train_transformer)
testset = datasets.ImageFolder(root="./dataset/", transform=test_transformer)

# 定义数据加载器
# trainset为数据集 batch_size为批大小 num_workers为线程数 shuffle为是否打乱 True为打乱
train_loader = DataLoader(trainset, batch_size=32, num_workers=0, shuffle=True)
test_loader = DataLoader(testset, batch_size=32, shuffle=True)


def train(model, train_loader, criterion, optimizer, num_epochs):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        # 训练时可看到对应的epoch和bach
        for inputs, lables in tqdm(train_loader, desc=f"epoch:{epoch+1}/{num_epochs}", unit="batch"):
            inputs, lables = inputs.to(device), lables.to(device)  # 将数据传到设备上
            optimizer.zero_grad()  # 梯度清零
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, lables)  # loss的计算
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            running_loss += loss.item() * inputs.size(0)  # 用loss×批次大小 得到该批次的loss
        epoch_loss = running_loss / \
            len(train_loader.dataset)  # 总损失除总数居大小 为我们每轮的损失
        print(f"epoch:{epoch+1}/{num_epochs} loss:{epoch_loss:.4f}")

        accuracy = evaluate(model, test_loader, criterion)
        if accuracy > best_acc:
            best_acc = accuracy
            save_model(model, save_path)
            print("model saved with best acc", best_acc)


def evaluate(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # 在评估模式下不需要计算梯度
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)  # 得到预测值
            total += labels.size(0)
            correct = correct + (predicted == labels).sum().item()  # 正确样本数累加
    avg_loss = test_loss / len(test_loader.dataset)
    accuracy = 100.0 * correct / total  # 计算准确率
    print(f"Test Loss:{avg_loss:.4f},Accuracy:{accuracy:.2f}%")
    return accuracy


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)  # 保存模型参数


if __name__ == "__main__":
    num_epochs = 10
    learning_rate = 0.001
    num_class = 4
    save_path = r"model_pth/best.pth"
    model = CNN(num_class).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train(model, train_loader, criterion, optimizer, num_epochs)
    evaluate(model, test_loader, criterion)
