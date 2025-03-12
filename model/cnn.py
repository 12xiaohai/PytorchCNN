import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_class):  # num_class为分类的个数
        super(CNN, self).__init__()
        # Sequential是一个有序的容器，神经网络模块将按照在传入Sequential的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数
        self.feature = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1,
                      padding=1),  # 保持图像大小不变 16*224*224 nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
            nn.ReLU(inplace=True),  # 卷积之后接上激活函数 增加非线性特征
            nn.MaxPool2d(kernel_size=2, stride=2),  # 池化之后变为16*112*112
            nn.Conv2d(16, 32, kernel_size=3, stride=1,
                      padding=1),  # 保持图像大小不变 32*112*112
            nn.ReLU(inplace=True),  # 卷积之后接上激活函数 增加非线性特征
            nn.MaxPool2d(kernel_size=2, stride=2),  # 池化之后变为32*56*56
        )
       # 定义全连接层 做分类
        self.classifier = nn.Sequential(
            nn.Dropout(),  # 防止过拟合
            nn.Linear(32 * 56 * 56, 128),  # 全连接层
            nn.ReLU(inplace=True),  # 激活函数
            nn.Dropout(),  # 防止过拟合
            nn.Linear(128, num_class),  # num_class为分类的个数
            nn.ReLU(inplace=True),  # 激活函数
        )

    def forward(self, x):
        # 前向传播部分
        x = self.feature(x)  # 提取特征
        x = x.view(x.size(0), -1)  # 展平 x.size(0)为batch_size
        x = self.classifier(x)  # 分类
        return x
