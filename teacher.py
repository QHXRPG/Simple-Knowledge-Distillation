import torch
import numpy
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
import torch.utils.data as Data
from torchvision.transforms import transforms

train_data = MNIST(root="/Users/qiuhaoxuan/PycharmProjects/pythonProject/机器学习/MNIST",
                   train=True,
                   transform=transforms.ToTensor(),
                   download=True)
train_loader = Data.DataLoader(dataset=train_data,batch_size=100,num_workers=0)
test_data = MNIST(root="/Users/qiuhaoxuan/PycharmProjects/pythonProject/机器学习/MNIST",
                  train=False,
                  transform=transforms.ToTensor(),
                  download=True)
test_loader = Data.DataLoader(dataset=test_data,batch_size=10000,num_workers=0)

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.l1 = nn.Linear(784,1000)
        self.l2 = nn.Linear(1000,1000)
        self.l3 = nn.Linear(1000,10)
        self.relu = nn.ReLU()
        self.d = nn.Dropout(p=0.5) #概率为0.5
    def forward(self,x):
        x=x.view(x.size(0),-1)
        x = self.l1(x)
        x = self.d(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.d(x)
        x = self.relu(x)
        x = self.l3(x)
        return x
net = net()

youhuaqi = torch.optim.Adam(net.parameters(), lr=0.005)
loss = nn.CrossEntropyLoss()
for epoch in range(25):
    for i,(x,y) in enumerate(train_loader):
        y_p = net(x)
        l = loss(y_p,y)
        youhuaqi.zero_grad()
        l.backward()
        youhuaqi.step()


for  i,(x,y)in enumerate(test_loader):
    if i>0:
        break

