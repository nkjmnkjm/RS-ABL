import os
import random
import shutil
import time
import pandas as pd

import os
import random
import re
import shutil

def shuff(image_dir,result_dir):
    # 1 读取文件列表
    # image_dir = r'image/' # 一开始存放有图片的文件夹，你需要修改成你的文件夹名字
    img_name_list = os.listdir(image_dir)
    print(img_name_list[:5])

    # 2 创建一个目标文件夹
    # result_dir = r'result/' # 目标文件夹（最终存放乱序后的文件夹），会自动创建
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f'创建文件夹{result_dir}成功！')

    # 3 创建随机数
    random_len = len(img_name_list)
    img_index = [i for i in range(random_len)]
    random.shuffle(img_index)
    # img_index

    # 开始转移每个文件
    for i, img in enumerate(img_name_list):
        dot_index = img.find('.')
        if dot_index > 0:
            img_name = str(img_index[i]) + img[dot_index:]
            shutil.copyfile(image_dir + img, result_dir + img.split('_')[0]+'_'+img_name) # 如果你的磁盘不够大，那么这个copyfile要改成move，但是此时需要注意备份好原来的文件夹里的文件！

def shuff1(image_dir,result_dir,typecsvfile,labelcsvfile,savefile,type=0):
    data1 = pd.read_csv(typecsvfile)
    namelist = data1.loc[:, 'class'].tolist()
    data2 = pd.read_csv(labelcsvfile)
    lablelist = []
    for i in range(data2.shape[0]):
        lablelist.append(data2.iloc[i, :].tolist())
    # 1 读取文件列表
    # image_dir = r'image/' # 一开始存放有图片的文件夹，你需要修改成你的文件夹名字
    img_name_list = os.listdir(image_dir)
    print(img_name_list[:5])

    # 2 创建一个目标文件夹
    # result_dir = r'result/' # 目标文件夹（最终存放乱序后的文件夹），会自动创建
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f'创建文件夹{result_dir}成功！')

    # 3 创建随机数
    random_len = len(img_name_list)
    img_index = [i for i in range(random_len)]
    random.shuffle(img_index)
    # img_index
    datalist=[]
    # 开始转移每个文件
    for i, img in enumerate(img_name_list):
        dot_index = img.find('.')
        if dot_index > 0:
            img_name = str(img_index[i]) + img[dot_index:]
            shutil.copyfile(image_dir + img, result_dir +img_name) # 如果你的磁盘不够大，那么这个copyfile要改成move，但是此时需要注意备份好原来的文件夹里的文件！
            a = img.split('_')[0]
            idx = namelist.index(a)
            if type == 0:
                datalist.append(lablelist[idx])
            else:
                lable = []
                for j in lablelist[idx]:
                    lable.append(j)
                lable.append(idx)
                datalist.append(lable)
    df = pd.DataFrame(datalist)
    a = data2.columns.tolist()
    if type != 0:
        a.append('class')
    df.columns = a
    df.to_csv(savefile, index=False, encoding='utf-8-sig')


def copyFile(fileDir, class_name):
    name = class_name
    #path = origion_path1
    image_list = os.listdir(fileDir)  # 获取图片的原始路径
    image_number = len(image_list)
    train_number = int(image_number * train_rate)
    train_sample = random.sample(image_list, train_number)  # 从image_list中随机获取0.75比例的图像.
    test_sample = list(set(image_list) - set(train_sample))
    sample = [train_sample, test_sample]

    # 复制图像到目标文件夹
    for k in range(len(save_dir)):
        if os.path.isdir(save_dir[k]):
            for name in sample[k]:
                name1 = name.split(".")[0] + '.xml'
                shutil.copy(os.path.join(fileDir, name), os.path.join(save_dir[k], name))
                #shutil.copy(os.path.join(path, name1), os.path.join(save_dir1[k], name1))
        else:
            os.makedirs(save_dir[k])
#            os.makedirs(save_dir1[k])
            for name in sample[k]:
                name1 = name.split(".")[0] + '.xml'
                shutil.copy(os.path.join(fileDir, name), os.path.join(save_dir[k], name))
                #shutil.copy(os.path.join(path, name1), os.path.join(save_dir1[k], name1))


def getcsv(filedir,typecsvfile,labelcsvfile,savefile,type=0):
    data1 = pd.read_csv(typecsvfile)
    namelist = data1.loc[:,'class'].tolist()
    data2 = pd.read_csv(labelcsvfile)
    lablelist = []
    for i in range(data2.shape[0]):
        lablelist.append(data2.iloc[i, :].tolist())
    image_list = os.listdir(filedir)
    datalist=[]
    for i in image_list:
        idx = namelist.index(i.split("_")[0])
        if type==0:
            datalist.append(lablelist[idx])
        else:
            lable = []
            for i in lablelist[idx]:
                lable.append(i)
            lable.append(idx)
            datalist.append(lable)
    df = pd.DataFrame(datalist)
    a = data2.columns.tolist()
    if type!=0:
        a.append('class')
    df.columns = a
    df.to_csv(savefile,index=False,encoding='utf-8-sig')

if __name__ == '__main__':
    # time_start = time.time()
    # # path = r"./多分类实验/所有数据"
    # path = './多分类实验/实验数据'
    # datanames = os.listdir(path)
    # for i in datanames:
    #     origion_path ='./多分类实验/实验数据'
    #     origion_path = origion_path+'/'+i
    #     #origion_path1 = './Annotations/'
    #
    #     # 保存路径
    #     save_train_dir = './多分类实验/50%实验/标记数据'+'/'+i
    #     save_test_dir = './多分类实验/50%实验/无标记数据'+'/'+i
    #     #save_train_dir1 = './train/Annotations/'
    #     #save_test_dir1 = './test/Annotations/'
    #     save_dir = [save_train_dir, save_test_dir]
    #     #save_dir1 = [save_train_dir1, save_test_dir1]
    #
    #     # 训练集比例
    #     train_rate = 0.5
    #    # 数据集类别及数量
    #     file_list = os.listdir(origion_path)
    #     num_classes = len(file_list)
    #     for i in range(num_classes):
    #         class_name = file_list[i]
    #     copyFile(origion_path, class_name)
    #     print('划分完毕!')
    #     time_end = time.time()
    #     print('---------------')
    #     print('训练集和测试集划分共耗时%s!' % (time_end - time_start))

# import pandas as pd
# from sklearn import model_selection
# df = pd.read_csv('./断案/1_0.90.csv')
# X = df.iloc[:,0:12]
# y = df.iloc[:,12]
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.05, random_state=0)
# a = pd.concat([X_train,X_test],axis=1)
# b = pd.concat([y_train,y_test],axis=1)


    # time_start = time.time()
    # # path = './多分类实验/train11'
    # path = './多分类实验/50%实验/标记数据'
    # datanames = os.listdir(path)
    # csvfile = './多分类实验/train7/lable.csv'
    # save_dir = './多分类实验/50%实验/有序'
    # for i in datanames:
    #     ordic = './多分类实验/50%实验/标记数据'+'/'+i
    #     imagelist = os.listdir(ordic)
    #     for j in imagelist:
    #         shutil.copy(os.path.join(ordic, j), os.path.join(save_dir, j))
    #     print('划分完毕!')
    #     time_end = time.time()
    #     print('---------------')
    #     print('训练集和测试集划分共耗时%s!' % (time_end - time_start))
    x=[3, 2, 12, 12, 102, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 24, 24, 114, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 36,
     36, 126, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 48, 48, 138, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 60, 60, 150,
     62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 72, 72, 162, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 84, 84, 174, 86, 86,
     86, 86, 86, 86, 86, 86, 86, 86, 96, 96, 186, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 108, 108, 198, 110, 110, 110,
     110, 110, 110, 110, 110, 110, 110, 120, 120, 210, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 132, 132, 222,
     134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 144, 144, 234, 146, 146, 146, 146, 146, 146, 146, 146, 146, 146,
     156, 156, 246, 158, 158, 158, 158, 158, 158, 158, 158, 158, 158, 168, 168, 258, 170, 170, 170, 170, 170, 170, 170,
     170, 170, 170, 180, 180, 270, 182, 182, 182, 182, 182, 182, 182, 182, 182, 182, 192, 192, 282, 194]
    y = [3, 2, 12, 12, 102, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 24, 24, 114, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
         36,
         36, 126, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38]
    x = range(0,len(y))
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))  # 定义图的大小
    plt.xlabel("time")     # X轴标签
    plt.ylabel("rule number")        # Y轴坐标标签
    plt.title("Number of rules in the knowledge base")      #  曲线图的标题
    plt.plot(x,y,color="red",linewidth=2)
    plt.savefig("number.png")
    # # shuff('./多分类实验/10%实验/有序/','./多分类实验/10%实验/乱序无标记/')
    # # getcsv('./多分类实验/10%实验/乱序无标记/','./AwA2-data/Animals_with_Attributes2/classes.csv','./AwA2-data/Animals_with_Attributes2/predicate-matrix-binary.csv','./多分类实验/label1.csv')
    # getcsv('./多分类实验/50%实验/序/','./AwA2-data/Animals_with_Attributes2/classes.csv','./AwA2-data/Animals_with_Attributes2/predicate-matrix-binary.csv','./多分类实验/50%实验/label2.csv',1)
    # shuff1('./多分类实验/10%实验/有序/','./多分类实验/10%实验/乱序无标记/','./AwA2-data/Animals_with_Attributes2/classes.csv','./AwA2-data/Animals_with_Attributes2/predicate-matrix-binary.csv','./多分类实验/10%实验/label1.csv')