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
import roughtset
from getnewrule import rule_tt
ImageSize = 32   # 定义图片大小
epoch = 30
train_dir = r"./多分类实验/10%实验/标记数据"    # 定义训练集存储路径
# train_dir = r"./多分类实验/train3"    # 定义训练集存储路径
# train_dir = r"./多分类实验/10%实验/无标记数据"
train_dir1 = r"./多分类实验/10%实验/无标记数据"
# train_dir1 ="./多分类实验/10%实验/乱序无标记"
test_dir = r"./多分类实验/测试数据"    # 定义测试集存储路径
# 定义数据集预处理方法
data_transform = transforms.Compose([
    transforms.Resize((ImageSize, ImageSize)),    # 重新调整图片大小
    transforms.ToTensor(),       # 将图片转换为Tensor数组
    # transforms.Normalize(mean=(0.5,0.5,0.5), std = (0.5,0.5,0.5))
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # transforms.Normalize((0.1307,), (0.3081,))		# 归一化操作
])
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
test_transform =  transforms.Compose([
    transforms.Resize((ImageSize, ImageSize)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
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
train_loader = data.DataLoader(datasets_train, batch_size=10, shuffle=True, num_workers=0)
train_loader1= data.DataLoader(datasets_train1,batch_size=10,shuffle=True,num_workers=0)
test_loader = data.DataLoader(datasets_test, batch_size=10, shuffle=True, num_workers=0)

#trainmodel1
# 构建CNN神经网络
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    # 构造方法，用于创建卷积神经网络需要使用到的模块
    def __init__(self):
        super(CNN, self).__init__()   # 继承父类的构造方法
        self.c1 = nn.Conv2d(3, 16, (3, 3))   # 定义第一个卷积层，输入3个节点输出16个节点，3*3的卷积
        self.c2 = nn.Conv2d(16, 32, (3, 3))      # 定义第二个卷积层，输入16个节点，输出32个节点，3*3的卷积
        self.c3 = nn.Conv2d(32, 64, (3, 3))     # 定义第三个卷积层，输入32个节点，输出64个节点，3*3的卷积
        self.b1 = nn.BatchNorm2d(16, affine=False)  # 定义第一个归一化层
        self.b2 = nn.BatchNorm2d(32, affine=False)  # 定义第二个归一化层
        self.b3 = nn.BatchNorm2d(64, affine=False)  # 定义第三个归一化层
        # self.fc1 = nn.Linear(331776, 32)      # 定义第一个全连接层
        self.fc1 = nn.Linear(4096, 32)
        self.fc2 = nn.Linear(32, 10)     # 定义第二个全连接层

    # 构建计算数据集中一张图片数据数量的函数
    def num_flat_features(self, x):
        """
        计算数据集中数据数量的函数
        :param x: 输入数据集
        :return:
        num_features：数据集中数据数量
        """
        size = x.size()[1:]  # 获取一个图片数据的大小
        num_features = 1  # 初始化数量
        for s in size:
            num_features *= s
        return num_features

    # 构建前向传播函数
    def forward(self, x):
        """
        构建前向传播函数
        :param x: 需要计算的数据
        :return:
            y_hat: 预测值
        """
        p1 = F.relu(self.c1(x))     # 第一个卷积层的运算，先卷积，然后激活
        b1 = self.b1(p1)    # 第一个归一化层运算
        p2 = F.relu(self.c2(b1))    # 第二个卷积层的运算，先卷积，然后激活
        b2 = self.b2(p2)    # 第二个归一化层运算
        p3 = F.relu(self.c3(b2))    # 第三个卷积层的运算，先卷积，然后激活
        b3 = self.b3(p3)    # 第三个归一化层运算
        m1 = F.max_pool2d(b3, 3, stride=3)    # 最大池化层，(3, 3)的卷积核，步长为3
        v1 = m1.view(-1, self.num_flat_features(m1))    # 将二维向量变为一维向量
        h1 = self.fc1(v1)   # 第一个全连接层
        y_hat = torch.sigmoid(self.fc2(h1))   # 第二个全连接层的计算
        # 返回预测值
        return y_hat
import torch.optim as optim
cnn = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on the test set: {100 * correct / total:.2f}%")
    return 100*correct/total
#
# 构建训练模型
# def trainModul(loader, module, criterion, optimizer, epochs=1):
#     """
#     构建训练模型
#     :param loader: 数据集
#     :param module: 训练模型
#     :param criterion: 损失函数
#     :param optimizer: 优化函数
#     :param epochs: 迭代轮数
#     :return:
#         model：训练后的模型
#     """
#     correct = 0  # 存储正确标签数
#     total = 0  # 存储标签总数
#     # 使用for循环进行迭代
#     for epoch in range(epochs):
#         for i, data in enumerate(loader):  # 使用for循环读取数据
#             x, y = data  # x:数据， y：真实值
#             y_hat = module(x)  # 前向传播，计算预测值
#             loss = criterion(y_hat, y)  # 计算损失值
#             optimizer.zero_grad()  # 清空导数
#             loss.backward()  # 后向传播
#             optimizer.step()  # 优化参数
#             predicted = torch.max(y_hat.data, 1)  # 选择最大概率的数输出
#             total += y.size(0)  # 计算总数
#             correct += (predicted.indices == y).sum()  # 累加正确标签数
#     #         if i % 10 == 9:  # 每迭代10次，输出一次损失值和准确率
#     #             print('[Epoch:%d, Batch:%5d] Loss:%.3f' % (epoch + 1, i + 1, loss))  # 输出损失值
#     #             print(correct)  # 输出正确标签数
#     #             print(total)  # 输出标签总数
#     #             print('ACCuracy on the test set: %0.3f %%' % (100 * correct / total))  # 输出准确率
#     # print('ACCuracy on the test set: %0.3f %%' % (100 * correct / total))  # 输出准确率
#     # print('Finished Training')
#     return module  # 返回训练好的模型
#
# def new_train(xlist,ylist,module, criterion, optimizer, epochs=1):
#     correct = 0  # 存储正确标签数
#     total = 0  # 存储标签总数
#     # 使用for循环进行迭代
#     module_old = module
#     for epoch in range(epochs):
#         for x,y in zip(xlist,ylist):
#             x=np.array(x)
#             y=np.array(y)
#             x = torch.from_numpy(x)
#             y = torch.from_numpy(y)
#             y_hat = module(x)  # 前向传播，计算预测值
#             loss = criterion(y_hat, y)  # 计算损失值
#             optimizer.zero_grad()  # 清空导数
#             loss.backward()  # 后向传播
#             optimizer.step()  # 优化参数
#             predicted = torch.max(y_hat.data, 1)  # 选择最大概率的数输出
#             total += y.size(0)  # 计算总数
#             correct += (predicted.indices == y).sum()  # 累加正确标签数
#     print('ACCuracy on the test set: %0.3f %%' % (100 * correct / total))  # 输出准确率
#     #print('Finished Training')
#
#     return module,(100 * correct / total)  # 返回训练好的模型
#
# # 构建测试函数
# def test(loader, model):
#     """
#     测试函数
#     :param loader: 测试数据集
#     :param model: 模型
#     :return:
#     """
#     correct = 0     # 存储正确标签数
#     total = 0   # 存储标签总数
#     for data in loader:
#         x, y = data # 获取数据及数据标签
#         y_hat = model(x)    # 获取预测概率
#         predicted = torch.max(y_hat.data, 1)    # 选择最大概率的数输出
#         total += y.size(0)  # 计算总数
#         correct += (predicted.indices == y).sum()   # 累加正确标签数
#     print('ACCuracy on the test set: %0.3f %%'%(100 * correct/total))  # 输出准确率
#     return 100 * correct/total

# cnn = CNN()	# 构建神经网络
# criterion = nn.CrossEntropyLoss()   # 定义损失函数
# optimizer = optim.SGD(cnn.parameters(), lr=0.0005, momentum=0.9)   # 定义优化函数

# 调用训练模型
old_mpdule = trainModul(train_loader, cnn, criterion, optimizer, epoch)
# 测试训练后的模型
maxcorr=test(test_loader, old_mpdule)

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
ruledict = {'top':[{'fly':['!=',1],'ans':['!=',1]}],'mid':[],'bot':[]}
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
n=5
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
    print('learn共耗时%s!' % (time_end - time_st

                           art))
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
# test1
# while True:
#     xlist=[]
#     ylist=[]
#     for i in oxlist:
#         xlist.append(i)
#     for i in oylist:
#         ylist.append(i)
#     for data in train_loader1:
#         x,y = data
#         y_hat = old_mpdule(x)
#         predicted = torch.max(y_hat.data,1)
#         fact2 = y.numpy()
#         fact1 = predicted.indices.numpy()
#         fact =np.concatenate(([fact2],[fact1]),axis=0)
#         fact = fact.T
#         n=swip.abuductionmodel(map={'0':[1],'1':[2,3],'2':[0]})
#
#         new_fact,dellist = n.abucemain(fact,y_hat.data.numpy())
#         fact = new_fact.T
#         tmpex = x.data.numpy()
#         tmpey = fact[1,:]
#
#         a = list(set(range(tmpey.shape[0]))-set(dellist))
#         tmpex = tmpex[a,:]
#         tmpey = tmpey[a]
#         if len(a)!=0:
#             xlist.append(tmpex)
#             ylist.append(tmpey)
#     mpdule,corr = new_train(xlist,ylist, cnn, criterion, optimizer, epochs=3)
#     new_map=map_train(map,xlist,ylist)
#     k=k+1
#     if k==50:
#         break
#     if corr >maxcorr:
#         old_mpdule = mpdule
#         maxcorr = corr
#         num=0
#     else:
#         num+=1
#         if num>5:
#             break
# test(test_loader, mpdule)

# test2
# import read
# from sklearn.cluster import KMeans
# train_dir1 ="./多分类实验/train4"
# while True:
#     xdatalist=[]
#     ydatalist=[]
#     for i in oxlist:
#         xdatalist.append(i)
#     for i in oylist:
#         ydatalist.append(i)
#     image1,image2 = read.readdata(train_loader1)
#     clt=KMeans(n_clusters=3)
#     clt.fit(image1)
#     kmlist = clt.labels_
#     xlist = []
#     ylist=  []
#     for k in range(len(image1)):
#         if len(list)<10:
#             xlist.append(image2[k])
#             ylist.append(kmlist[k])
#         else:
#             x= np.array(xlist)
#             y_hat = old_mpdule(torch.from_numpy(x))
#             predicted = torch.max(y_hat.data, 1)
#             fact1 = predicted.indices.numpy()
#             fact2 = np.array(ylist)
#             fact = np.concatenate(([fact2], [fact1]), axis=0)
#             fact = fact.T
#             map = {'0': [1], '1': [2, 3], '2': [0]}
#             n=swip.abuductionmodel(map=map)
#             keylist = list(map.keys())
#             new_fact, dellist,map = n.abucemain(fact, y_hat.data.numpy(),len(keylist),4)
#             fa = new_fact.T
#             tmpex = x
#             tmpey = fact[1,:]
#
#             a = list(set(range(tmpey.shape[0]))-set(dellist))
#             tmpex = tmpex[a,:]
#             tmpey = tmpey[a]
#             if len(a)!=0:
#                 xdatalist.append(tmpex)
#                 ydatalist.append(tmpey)
#     mpdule,corr = new_train(xdatalist,ydatalist, cnn, criterion, optimizer, epochs=3)
#     k=k+1
#     if k==50:
#         break
#     if corr >maxcorr:
#         old_mpdule = mpdule
#         maxcorr = corr
#         num=0
#     else:
#         num+=1
#         if num>5:
#             break
# test(test_loader, mpdule)