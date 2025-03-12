import torch
from model.cnn import CNN

x = torch.randn(32, 3, 224, 224)
model = CNN(num_class=4)
y = model(x)
print(y.shape)
