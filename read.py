import os
import numpy as np
from sklearn.cluster import KMeans
import cv2
from imutils import build_montages
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from sklearn.metrics import fowlkes_mallows_score
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(resnet50.conv1,
                                    resnet50.bn1,
                                    resnet50.relu,
                                    resnet50.maxpool,
                                    resnet50.layer1,
                                    resnet50.layer2,
                                    resnet50.layer3,
                                    resnet50.layer4)

    def forward(self, x):
        x = self.resnet(x)
        return x

def readdata(path):
    net = Net().eval()

    image_path = []
    all_images1 = []
    all_images2=[]
    images = os.listdir(path)

    for image_name in images:
        image_path.append(path+'/' + image_name)
    for path in image_path:
        image = Image.open(path).convert('RGB')
        image = transforms.Resize([224,224])(image)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize((0.1307,), (0.3081,))(image)
        all_images2.append(image)
        image = image.unsqueeze(0)
        image = net(image)
        image = image.reshape(-1, )
        all_images1.append(image.detach().numpy())
    return all_images1,all_images2
#
# clt = KMeans(n_clusters=4)
# clt.fit(l_images)
# labelIDs = np.unique(clt.labels_)
from sklearn.metrics import calinski_harabaz_score
def chtest(data,label):
    score = calinski_harabaz_score(data, label)
    print('数据聚类评价分值为：%f' % ( score))
    return score

net = Net().eval()

image_path = []
all_images = []
images = os.listdir('./多分类实验/train4')

for image_name in images:
    image_path.append('./多分类实验/train4/' + image_name)
for path in image_path:
    image = Image.open(path).convert('RGB')
    image = transforms.Resize([224,224])(image)
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    image = net(image)
    image = image.reshape(-1, )
    all_images.append(image.detach().numpy())
scorelist=[]
for i in range(2,7):
    kmeans = KMeans(n_clusters=i).fit(all_images)
    score = chtest(all_images,kmeans.labels_)
    scorelist.append(score)
print(scorelist)
