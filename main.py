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
    attribute_path = r"attribute.csv"
    path = r"/home/slam/PycharmProjects/rough-ABL/AwA2-data/JPEGImages"
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


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    base_model = VGG19(weights='imagenet', include_top=True)
    model_fc1 = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)  # fc1 layer

    class_list, class_num, class_att, image_path = get_files()  ##获取图片路径列表
    print(class_list)
    print(class_num)
    print(class_att)

    fc1s = None
    atts = None

    for i, path in enumerate(image_path):
        print('number ' + str(i) + ' path: ' + path)
        try:
            img = image.load_img(path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            fc1 = model_fc1.predict(x)
            key = str.split(str.split(path, '\\')[-1], '_')[0]
            att = class_att[key]
            if fc1s is None:
                fc1s = fc1
                atts = att
            else:
                fc1s = np.row_stack((fc1s, fc1))
                atts = np.row_stack((atts, att))
        except IOError:
            print('########################Image Error###########################')
    # save extracted feature
    sio.savemat('vgg19_feature', {'fc1': fc1s, 'atts': atts})
    fw = open("animals_num.txt", 'w+')
    fw.write(str(class_num))  # 把字典转化为str
    fw.close()
    fw = open("animals_sequence.txt", 'w')
    fw.write(str(class_list))  # 把list转化为str
    fw.close()