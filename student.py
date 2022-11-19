import torch
import numpy
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
import torch.utils.data as Data
from torchvision.transforms import transforms
import torch.nn.functional as F
train_data = MNIST(root="/Users/qiuhaoxuan/PycharmProjects/pythonProject/机器学习/MNIST",
                   train=True,
                   transform=transforms.ToTensor(),
                   download=True)
train_loader = Data.DataLoader(dataset=train_data,batch_size=60,num_workers=0)
test_data = MNIST(root="/Users/qiuhaoxuan/PycharmProjects/pythonProject/机器学习/MNIST",
                  train=False,
                  transform=transforms.ToTensor(),
                  download=True)
test_loader = Data.DataLoader(dataset=test_data,batch_size=10000,num_workers=0)


#学生网络
class student_net(nn.Module):
    def __init__(self):
        super(student_net, self).__init__()
        self.l1 = nn.Linear(784,20)
        self.l2 = nn.Linear(20,20)
        self.l3 = nn.Linear(20,10)
        self.relu = nn.ReLU()
        self.d = nn.Dropout(p=0.5) #概率为0.5
    def forward(self,x):
        x=x.view(x.size(0),-1)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        return x

#老师网络
class teacher_net(nn.Module):
    def __init__(self):
        super(teacher_net, self).__init__()
        self.l1   = nn.Linear(784,1000)
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

teacher = teacher_net()
student = student_net()
teacher.load_state_dict(torch.load("/Users/qiuhaoxuan/PycharmProjects/pythonProject/机器学习/知识蒸馏/teacher.pth"))
teacher.eval()

hard_loss = nn.CrossEntropyLoss()
soft_loss = nn.KLDivLoss(reduction="batchmean") #不然loss不仅会在batch维度上取平均,还会在概率分布的维度上取平均
youhuaqi = torch.optim.Adam(student.parameters(), lr=0.005)
T = 7
z=0
for epoch in range(25):
    for i,(x,y) in enumerate(train_loader):
        teacher.zero_grad()
        teacher_p = teacher(x)
        student_p = student(x)
        hardloss = hard_loss(student_p,y)
        softloss = soft_loss(F.softmax(student_p/7,dim=1),
                             F.softmax(teacher_p/7,dim=1))   #按列进行softmax
        Loss = 0.3*hardloss +0.7*softloss
        z=z+1
        if z==300:
            print(Loss,hardloss,softloss)
            z=0
        youhuaqi.zero_grad()
        Loss.backward()
        youhuaqi.step()


for ii,(x_t,y_t) in enumerate(test_loader):
    if ii>0:
        break

c = torch.argmax(student(x_t),1)

