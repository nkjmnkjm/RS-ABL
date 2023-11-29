import torch
import torch.nn as nn
from torch.nn import functional as F
# 导入包
import os		# 用于当前文件
# 用于构建神经网络
import time
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
import swip
import numpy as np
from getnewrule import rule_tt
ImageSize = 32    # 定义图片大小
epoch = 3
train_dir = r"./多分类实验/50%实验/标记数据"    # 定义训练集存储路径
# train_dir = r"./多分类实验/train3"    # 定义训练集存储路径
train_dir = r"./多分类实验/10%实验/无标记数据"
# train_dir = r"./多分类实验/实验数据"
# train_dir1 = r"./多分类实验/50%实验/无标记数据"
train_dir1 ="./多分类实验/10%实验/无标记数据"
test_dir = r"./多分类实验/测试数据"    # 定义测试集存储路径
# 定义数据集预处理方法
# data_transform = transforms.Compose([
#     transforms.Resize((ImageSize, ImageSize)),    # 重新调整图片大小
#     transforms.ToTensor(),       # 将图片转换为Tensor数组
#     # transforms.Normalize(mean=(0.5,0.5,0.5), std = (0.5,0.5,0.5))
#     transforms.Normalize((0.1307,), (0.3081,))		# 归一化操作
# ])
# data_transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomCrop(32, padding=4),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
# data_transform= transforms.Compose([
#     transforms.Resize((224, 224)),  # 先调整图片大小至256x256
#     transforms.RandomCrop((224, 224)),  # 再随机裁剪到224x224
#     transforms.RandomHorizontalFlip(),  # 随机的图像水平翻转，通俗讲就是图像的左右对调
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 归一化，数值是用ImageNet给出的数值
# ])
data_transform = transforms.Compose(
    [
        transforms.Resize((ImageSize, ImageSize))   ,
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
test_transform =  transforms.Compose([
    transforms.Resize((ImageSize, ImageSize)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
])

# datasets.ImageFolder(dir, transform)：导入数据集
# dir：数据集存储路径
# transform：数据集预处理方法
datasets_train = datasets.ImageFolder(train_dir, data_transform)
datasets_train1 = datasets.ImageFolder(train_dir1, data_transform)
datasets_test = datasets.ImageFolder(test_dir, test_transform)

# data.DataLoader(datasets, batch_size, shuffle=True, num_workers=0)：加载数据集
# datasers：导入的数据集
# batch_size：每次调用的图片数量
train_loader = data.DataLoader(datasets_train, batch_size=100, shuffle=True, num_workers=0)
train_loader1= data.DataLoader(datasets_train1,batch_size=100,shuffle=True,num_workers=0)
test_loader = data.DataLoader(datasets_test, batch_size=50, shuffle=True, num_workers=0)


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual
    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)
class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, activate_before_residual))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)
class WideResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                import math
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

class RestNet18(nn.Module):
    def __init__(self):
        super(RestNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
                                    RestNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [2, 1]),
                                    RestNetBasicBlock(256, 256, 1))

        self.layer4 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]),
                                    RestNetBasicBlock(512, 512, 1))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out

def trainModul(trainloader,net,criterion,optimizer,num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        total = 0
        for i, data in enumerate(trainloader,0):
            inputs, labels = data
            labels1 = labels.to(device)
            labels = F.one_hot(labels,10)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            running_corrects += (predicted.indices == labels1).sum().item()
            running_loss += loss.item()

        scheduler.step()
        epoch_loss = running_loss / (i + 1)
        epoch_acc = running_corrects / total
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    print("Training finished.")
    return net
def new_train(xlist,ylist,net, criterion, optimizer, epochs=1):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # net.to(device)
    for epoch in range(epochs):
        running_loss = 0.0
        running_corrects = 0
        total = 0
        i = 0
        for x, y in zip(xlist, ylist):
            x = np.array(x)
            y = np.array(y)
            x = torch.from_numpy(x)
            y = torch.from_numpy(y)
            y1 = y.to(device)
            y = F.one_hot(y,10)
            inputs, labels = x.to(device), y.to(device)
            # inputs = x
            # labels = y

            outputs = net(inputs)
            loss = criterion(outputs, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            running_corrects += (predicted.indices == y1).sum().item()
            running_loss += loss.item()
            i = i+1

        scheduler.step()
        epoch_loss = running_loss / (i + 1)
        epoch_acc = running_corrects / total
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    print("Training finished.")
    return net,epoch_loss

def test(testloader,net):
    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    top5correct=0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            maxk = max((1, 5))
            y_resize = labels.view(-1, 1)
            _, pred = outputs.topk(maxk, 1, True, True)
            top5correct += torch.eq(pred, y_resize).sum().float().item()

    print(f"Accuracy on the test set: {100 * correct / total:.2f}%")
    print(f"Top5 Accuracy on the test set: {100 * top5correct / total:.2f}%")
    return 100*correct/total,100 * top5correct / total
# cnn = RestNet18()
acclist=[]
# for n in range(5,60,5):
cnn = WideResNet(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
old_mpdule = trainModul(train_loader, cnn, criterion, optimizer, epoch)
# 测试训练后的模型
maxcorr,_=test(test_loader, old_mpdule)

k = 0
num=0
oxlist=[]
oylist=[]
for data in train_loader:
    x,y = data
    tmpy = y.numpy()
    tmpx = x.data.numpy()
    oxlist.append(tmpx)
    oylist.append(tmpy)
    # if num==0:
    #     xdata = tmpx
    #     ydata = tmpy
    # else:
    #     xdata = np.concatenate(([xdata],[tmpx]),axis=0)
    #     ydata = np.concatenate(([ydata],[tmpy]),axis=0)


# test1
num=0
csvfile = './多分类实验/50%实验/label2.csv'
import pandas as pd
from abuce_judge import new_abuce
data1 = pd.read_csv(csvfile)
# ruledict = {'top':[{'fly':['!=',1],'ans':['!=',1]}],'mid':[],'bot':[]}
ruledict = {'top':[{'flys':['!=',1],'ans':['!=',1]},{'flys':['!=',1],'swims':['!=',0],'ans':['!=',1]},{'flys':['==',0],'ans':['==',1]}],'mid':[],'bot':[]}
# namelist = data1.columns.tolist()[0:8]
namelist = ['furry','big','paws','tail','meatteeth','flys','swims','forager','hunter','horns','tusks']
# namelist = ['spots','stripes','furry']
xdata = data1.loc[:,namelist]
ydata = data1.loc[:,'class'].tolist()
clalldic = {'0':0,'2':6,'6':5,'18':4,'29':1,'33':8,'41':7,'46':9,'48':2,'49':3}
newy=[]
pclalldic = {'0':0,'6':2,'5':6,'4':18,'1':29,'8':33,'7':41,'9':46,'2':48,'3':49}
for i in ydata:
    if str(i) in list(clalldic.keys()):
        newy.append(clalldic[str(i)])
ydata = newy
rulelen = []
# anslist = []
# classlist = []
# for i in range(len(ydata)):
#     if ydata[i] not in classlist:
#         classlist.append(ydata[i])
#         anslist.append(xdata.loc[i,:])
# datakind = pd.DataFrame(anslist)
rule_treat = rule_tt(xdata,ydata)
a=0
for key,item in ruledict.items():
    if len(ruledict[key])!=0:
        a+=len(ruledict[key])
        ruledict[key] = rule_treat.rule_reduction(ruledict[key],0.9)
rulelen.append(a)
a = 0
for key,item in ruledict.items():
    if len(ruledict[key])!=0:
        a = a+len(ruledict[key])
rulelen.append(a)
result1,result2,nondic,bonddic,maprulelist = rule_treat.rule_gen()
result1 = rule_treat.rule_reduction(result1)
rulelen.append(a+len(result1))
result1 = rule_treat.rule_merge(result1)
for i in result1:
    ruledict['mid'].append(i)
a = 0
for key,item in ruledict.items():
    if len(ruledict[key])!=0:
        a = a+len(ruledict[key])
rulelen.append(a)
result1,result2,nondic,bonddic,maprulelist = rule_treat.norule_gen()
# rulelen.append(a+len(result1))
result1 = rule_treat.rule_reduction(result1)
result1 = rule_treat.rule_merge(result1)
for i in result1:
    ruledict['mid'].append(i)
a = 0
for key, item in ruledict.items():
  if len(ruledict[key]) != 0:
        a = a + len(ruledict[key])
rulelen.append(a)
Num=0
n=15

# datalable = pd.read_csv('./多分类实验/10%实验/label1.csv')
# datalable = datalable.loc[:,namelist]
labelcsvfile = './AwA2-data/Animals_with_Attributes2/predicate-matrix-binary.csv'
data2 = pd.read_csv(labelcsvfile)
lablelist = []
for i in range(data2.shape[0]):
    lablelist.append(data2.iloc[i, :].tolist())
while Num<n:
    xlist = []
    ylist = []
    for i in oxlist:
        xlist.append(i)
    for i in oylist:
        ylist.append(i)
    tmpnum = 0
    time_start = time.time()
    for data in train_loader1:
        x, y = data
        yo = y
        y = F.one_hot(y,10)
        x1,y1 = x.to(device), y.to(device)
        y_hat = old_mpdule(x1)
        predicted = torch.max(y_hat.data, 1)[1].cpu().numpy()
        # fact1 = list(predicted.indices.numpy())
        fact1 = list(predicted)
        abucemodel = new_abuce(ruledict)
        # if tmpnum*10+10<datalable.shape[0]:
        #     datax = datalable.loc[tmpnum*10:(tmpnum+1)*10-1,:]
        # else:
        #     datax = datalable.loc[tmpnum*10:datalable.shape[0]-1,:]
        ytmp = yo.numpy().tolist()
        datax = []
        for i in ytmp:
            if str(i) in list(pclalldic.keys()):
                datax.append(lablelist[pclalldic[str(i)]])
        datax = pd.DataFrame(datax)
        datax.columns = data2.columns.tolist()
        datax = datax.loc[:,namelist]
        #
        # datax = datax.reset_index(drop=True)
        tmpnum = tmpnum+1
        new_fact, dellist,ruledict= abucemodel.abuce(datax, fact1)
        a = 0
        for key, item in ruledict.items():
            if len(ruledict[key]) != 0:
                a = a + len(ruledict[key])
        rulelen.append(a)
        tmpex = x.data.numpy()
        tmpey = new_fact
        a = list(set(range(len(tmpey)))-set(dellist))
        tmpex = tmpex[a,:]
        alist=[]
        for idx in a:
            alist.append(tmpey[idx])
        tmpey = alist
        if len(a)!=0:
            xlist.append(tmpex)
            ylist.append(tmpey)
    # a = 0
    # for key, item in ruledict.items():
    #     if len(ruledict[key]) != 0:
    #         a = a + len(ruledict[key])
    #         ruledict[key] = rule_treat.rule_reduction(ruledict[key], 0.9)
    # rulelen.append(a)
    # result1, result2, nondic, bonddic, maprulelist = rule_treat.rule_gen()
    # result1 = rule_treat.rule_reduction(result1)
    # rulelen.append(a + len(result1))
    # result1 = rule_treat.rule_merge(result1)
    # for i in result1:
    #     ruledict['mid'].append(i)
    # a = 0
    # for key, item in ruledict.items():
    #     if len(ruledict[key]) != 0:
    #         a = a + len(ruledict[key])
    # rulelen.append(a)
    # result1, result2, nondic, bonddic, maprulelist = rule_treat.norule_gen()
    # rulelen.append(a + len(result1))
    # result1 = rule_treat.rule_reduction(result1)
    # result1 = rule_treat.rule_merge(result1)
    # for i in result1:
    #     ruledict['mid'].append(i)
    # a = 0
    # for key, item in ruledict.items():
    #     if len(ruledict[key]) != 0:
    #         a = a + len(ruledict[key])
    # rulelen.append(a)
    time_end = time.time()
    print('aabuce共耗时%s!' % (time_end - time_start))
    time_start = time.time()

    old_mpdule, corr = new_train(xlist, ylist, old_mpdule, criterion, optimizer,epoch)
    time_end = time.time()
    print('learn共耗时%s!' % (time_end - time_start))
    Num = Num+1
    test(test_loader, old_mpdule)
acc,_=test(test_loader, old_mpdule)
acclist.append(acc)
print(acc)
print(rulelen)
# x=range(0,len(rulelen[0:-7]))
# y = rulelen
# import matplotlib.pyplot as plt
# plt.plot(x,y,color="red",linewidth=2)
# plt.figure(figsize=(8,6))  # 定义图的大小
# plt.xlabel("time(s)")     # X轴标签
# plt.ylabel("rule number")        # Y轴坐标标签
# plt.title("Number of rules in the knowledge base")      #  曲线图的标题
# plt.savefig("number.png")