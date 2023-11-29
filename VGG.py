
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import os
import ssl
import numpy as np
from getnewrule import rule_tt
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
ImageSize = 32    # 定义图片大小
epoch = 30
train_dir = r"./多分类实验/10%实验/标记数据"    # 定义训练集存储路径
# train_dir = r"./多分类实验/train3"    # 定义训练集存储路径
# train_dir = r"./多分类实验/10%实验/无标记数据"
train_dir1 = r"./多分类实验/10%实验/无标记数据"
# train_dir1 ="./多分类实验/10%实验/乱序无标记"
test_dir = r"./多分类实验/测试数据"    # 定义测试集存储路径
# 定义数据集预处理方法
data_transform = transforms.Compose(
    [
        transforms.Resize((ImageSize, ImageSize))   ,
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_transform =  transforms.Compose([
    transforms.Resize((ImageSize, ImageSize))   ,
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
train_loader = data.DataLoader(datasets_train, batch_size=64, shuffle=True, num_workers=0)
train_loader1= data.DataLoader(datasets_train1,batch_size=64,shuffle=True,num_workers=0)
test_loader = data.DataLoader(datasets_test, batch_size=50, shuffle=True, num_workers=0)
classes = ('antelope', 'bat', 'cow', 'dolphin', 'elephant', 'horse', 'killer+whale', 'pig','rat','walrus')


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, 128, 1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.conv8 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv10 = nn.Conv2d(256, 256, 1, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        self.conv11 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv13 = nn.Conv2d(512, 512, 1, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()

        self.fc14 = nn.Linear(512 * 4 * 4, 1024)
        self.drop1 = nn.Dropout2d()
        self.fc15 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout2d()
        self.fc16 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        # print(" x shape ",x.size())
        x = x.view(-1, 512 * 4 * 4)
        x = F.relu(self.fc14(x))
        x = self.drop1(x)
        x = F.relu(self.fc15(x))
        x = self.drop2(x)
        x = self.fc16(x)

        return x
def trainModul(trainloader,net,criterion,optimizer,num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        total = 0
        for i, data in enumerate(trainloader,0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            running_corrects += (predicted.indices == labels).sum().item()
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
            inputs, labels = x.to(device), y.to(device)
            # inputs = x
            # labels = y

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            running_corrects += (predicted.indices == labels).sum().item()
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
    top5correct = 0
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
cnn = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
old_mpdule = trainModul(train_loader, cnn, criterion, optimizer, epoch)
# 测试训练后的模型
maxcorr,top5corr=test(test_loader, old_mpdule)

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
csvfile = './多分类实验/10%实验/label2.csv'
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
# anslist = []
# classlist = []
# for i in range(len(ydata)):
#     if ydata[i] not in classlist:
#         classlist.append(ydata[i])
#         anslist.append(xdata.loc[i,:])
# datakind = pd.DataFrame(anslist)
rule_treat = rule_tt(xdata,ydata)
for key,item in ruledict.items():
    if len(ruledict[key])!=0:
        ruledict[key] = rule_treat.rule_reduction(ruledict[key],0.9)
result1,result2,nondic,bonddic,maprulelist = rule_treat.rule_gen()
result1 = rule_treat.rule_reduction(result1)
result1 = rule_treat.rule_merge(result1)
for i in result1:
    ruledict['mid'].append(i)
result1,result2,nondic,bonddic,maprulelist = rule_treat.norule_gen()
result1 = rule_treat.rule_reduction(result1)
result1 = rule_treat.rule_merge(result1)
for i in result1:
    ruledict['mid'].append(i)
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
        ytmp = y.numpy().tolist()
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
    time_end = time.time()
    print('aabuce共耗时%s!' % (time_end - time_start))
    time_start = time.time()
    mpdule, corr = new_train(xlist, ylist, old_mpdule, criterion, optimizer,epoch)
    time_end = time.time()
    print('learn共耗时%s!' % (time_end - time_start))
    if corr >maxcorr:
        old_mpdule = mpdule
        maxcorr = corr
        num=0
    else:
        num+=1
        if num>5:
            break
    Num = Num+1
    test(test_loader, mpdule)
test(test_loader, mpdule)