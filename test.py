# import os
# import numpy as np
# from sklearn.cluster import KMeans
# import cv2
# from imutils import build_montages
# import torch.nn as nn
# import torchvision.models as models
# from PIL import Image
# from torchvision import transforms
#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         resnet50 = models.resnet50(pretrained=True)
#         self.resnet = nn.Sequential(resnet50.conv1,
#                                     resnet50.bn1,
#                                     resnet50.relu,
#                                     resnet50.maxpool,
#                                     resnet50.layer1,
#                                     resnet50.layer2,
#                                     resnet50.layer3,
#                                     resnet50.layer4)
#
#     def forward(self, x):
#         x = self.resnet(x)
#         return x
#
# net = Net().eval()
#
# image_path = []
# all_images = []
# images = os.listdir('./多分类实验/train4')
#
# for image_name in images:
#     image_path.append('./多分类实验/train4/' + image_name)
# for path in image_path:
#     image = Image.open(path).convert('RGB')
#     image = transforms.Resize([224,224])(image)
#     image = transforms.ToTensor()(image)
#     image = image.unsqueeze(0)
#     image = net(image)
#     image = image.reshape(-1, )
#     all_images.append(image.detach().numpy())
#
# clt = KMeans(n_clusters=4)
# clt.fit(all_images)
# labelIDs = np.unique(clt.labels_)
# print(labelIDs)
# print(np.where(clt.labels_ == 0)[0])
# # for labelID in labelIDs:
# # 	idxs = np.where(clt.labels_ == labelID)[0]
# # 	idxs = np.random.choice(idxs, size=min(25, len(idxs)),replace=False)
# # 	show_box = []
# # 	for i in idxs:
# # 		image = cv2.imread(image_path[i])
# # 		image = cv2.resize(image, (96, 96))
# # 		show_box.append(image)
# # 	montage = build_montages(show_box, (96, 96), (5, 5))[0]
# #
# # 	title = "Type {}".format(labelID)
# # 	cv2.imshow(title, montage)
# # 	cv2.waitKey(0)

# -*- coding: utf-8 -*-
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import os
import numpy as np
import scipy.io as sio
import csv


##获取文件路径列表dir+filename
def get_files():
    attribute_path = r"/home/slam/PycharmProjects/rough-ABL/attribute.csv"
    path = r"/home/slam/PycharmProjects/rough-ABL/AwA2-data/Animals_with_Attributes2/JPEGImages"
    class_list = []
    class_num = {}
    class_att = {}
    image_path = []
    image_att = None

    attribute_file = csv.reader(open(attribute_path))
    # get attribute
    print('get attribute')
    for row in attribute_file:
        att = [float(x) for x in row[1:]]
        class_att[row[0]] = att
        class_num[row[0]] = 0

    for root, dirs, files in os.walk(path):
        for name in files:
            animal = str.split(name, '_')[0]
            if animal not in class_list:
                class_list.append(animal)
            class_num[animal] = class_num[animal] + 1
            if image_att is None:
                image_att = class_att[animal]
            else:
                image_att = np.row_stack((image_att, class_att[animal]))
            image_path.append(os.path.join(root, name))
    return class_list, class_num, class_att, image_path

def getdata():
    class_list = []
    class_num = {}
    class_att = {}
    image_path = []
    image_att = None
    path = r"/home/slam/PycharmProjects/rough-ABL/AwA2-data/Animals_with_Attributes2/JPEGImages"
    for root, dirs, files in os.walk(path):
        for name in files:
            animal = str.split(name, '_')[0]
            if animal not in class_list:
                class_list.append(animal)
            class_num[animal] = class_num[animal] + 1
            if image_att is None:
                image_att = class_att[animal]
            else:
                image_att = np.row_stack((image_att, class_att[animal]))
            image_path.append(os.path.join(root, name))
