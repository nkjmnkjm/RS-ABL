import pandas as pd
from roughtset import roughchange,roughset
from abuce_judge import abuce_model
from train_model import predict_model
import csv
from sklearn.utils import shuffle
from getnewrule import rule_tt
import json
import sklearn
import copy
import numpy as np


def parse_condition(rule):
    ret = []
    for r in rule.split("^"):
        d = r.split(':')
        ret.append((int(d[0]), int(d[1])))
    return ret

def readrule(rule_file):
    rule_list = []
    with open(rule_file) as fin:
        for rule in fin:
            rule = rule.strip().replace(" ", "").split('#')
            name,left, right, p = rule
            print(rule)
            print(p)
            name = parse_condition(name)
            left = parse_condition(left)
            right = parse_condition(right)
            rule_list.append((name,left, right, float(p)))
    return rule_list


def readfile(csv_file):
    with open(csv_file, 'r', encoding="utf-8") as f:
        data = []
        labels1 = []
        labels2 = []
        filenames = []
        ahs = []
        attrs = []

        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            [filename, ah, money, damage, attitude, surrender, again, young, forgive, tool, room, theft, year_num,probation, money_num] = row
            if float(money) < 0.1:
                continue
            if int(year_num) == 0:
                continue
            if float(money) >= 30000:
                continue
            money = process_money(float(money), 1000)
            data.append([money])
            attrs.append(
                [int(damage), int(attitude), int(surrender), int(again), int(young), int(forgive), int(tool), int(room),
                 int(theft)])
            labels1.append(int(year_num))
            labels2.append(int(probation))
            filenames.append(filename)
            ahs.append(ah)

    filenames, ahs, data, labels1,labels2, attrs = shuffle(filenames, ahs, data, labels1,labels2, attrs, random_state=5)
    return filenames, ahs, data, labels1,labels2, attrs

def dfutil(df,type=0):
    alist = []

    for i in range(df.shape[0]):
        if df.shape[1]<2:
            if type==1:
                alist.append([int(df.iloc[i,:])])
            else:
                alist.append(int(df.iloc[i, :]))
        else:
            blist = []
            for j in df.iloc[i,:].tolist():
                blist.append(int(j))
            alist.append(blist)
    return alist

def process_money(money,seg):
    if money >= seg:
        m = (money + seg / 2) // seg * seg
    elif money <= 499:
        m = 499
    elif money <= 999:
        m = 999
    else:
       m = money
    return m

if __name__=='__main__':
    # filepath = './断案/0_0.10.csv'
    # rulepath = './断案/rule_file.txt'
    # filenames, ahs, data, labels1,labels2, attrs = readfile(filepath)
    # predictmodel = predict_model()
    # abucemodel = abuce_model(rulepath)
    # predictmodel.fit(data,attrs,labels1,5)
    # data = pd.read_csv(filepath)
    # data1 = data.iloc[:,3:12]
    # title = data1.columns.values.tolist()
    # data2 = data.iloc[:,12].to_frame()
    # rough_model = roughset(data1,data2)
    # a = rough_model.deal()
    # rough = a[0]
    # abl_Time = 3
    #
    # filepath = './断案/0_0.09.csv'
    # filenames, ahs, data, labels1,labels2, attrs = readfile(filepath)
    # testfilepath = './断案/1_0.90.csv'
    # filenames1, ahs1, data1, labels11,labels21, attrs1 = readfile(testfilepath)

    # rule_treat = rule_tt(X_train,y_train)
    # result1, result2, nondic, bonddic, maprulelist = rule_treat.rule_gen()
    # result1 = rule_treat.rule_reduction(result1)
    # result1 = rule_treat.rule_merge(result1)
    # for i in result1:
    #     ruledict['mid'].append(i)
    # result1, result2, nondic, bonddic, maprulelist = rule_treat.norule_gen()
    # # rulelen.append(a+len(result1))
    # result1 = rule_treat.rule_reduction(result1)
    # result1 = rule_treat.rule_merge(result1)
    # for i in result1:
    #     ruledict['mid'].append(i)
    # from abuce_judge import new_abuce
    # abucemodel1 = new_abuce(ruledict)

    rulepath = './断案/rule_file.txt'
    abucemodel = abuce_model(rulepath)
    #filepath = './断案/1_0.90.csv'
    filepath = './断案/0_0.10.csv'
    df = pd.read_csv(filepath)
    from sklearn import model_selection

    X_train = df.iloc[:, 2:12]
    y_train = df.iloc[:, 12]
    ruledict = {'top':[],'mid':[],'bot':[]}
    #X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)
    filepath = './断案/10.csv'
    df = pd.read_csv(filepath)
    X_test = df.iloc[:, 2:12]
    y_test = df.iloc[:, 12]
    moneys = dfutil(X_train.iloc[:, 0].to_frame(), 1)
    moneyslist = []
    for i in moneys:
        moneyslist.append([process_money(i[0], 1000)])
    moneys = moneyslist
    title = X_train.iloc[:, 1:10].columns.values.tolist()
    attr = dfutil(X_train.iloc[:, 1:10])
    mouth = dfutil(y_train.to_frame())
    predictmodel = predict_model()
    predictmodel.fit(moneys, attr, mouth, 3)
    moneys = dfutil(X_test.iloc[:, 0].to_frame(), 1)
    data1 = []
    for i in moneys:
        data1.append([process_money(i[0], 1000)])
    attrs1 = dfutil(X_test.iloc[:, 1:10])
    labels11 = dfutil(y_test.to_frame())
    mae, mse, percent, perc_cnt = predictmodel.test(data1, attrs1, labels11)
    print('*************')
    print(mae)
    print(mse)
    print(percent)
    print(perc_cnt)
    print('************')
    # X_train1, X_test1, y_train1, y_test1 = model_selection.train_test_split(X_train, y_train, test_size=0.9)
    X_train1 = X_train
    y_train1 = y_train

    moneys = dfutil(X_train1.iloc[:,0].to_frame(),1)
    moneyslist=[]
    for i in moneys:
        moneyslist.append([process_money(i[0],1000)])
    moneys = moneyslist
    title = X_train1.iloc[:,1:10].columns.values.tolist()
    attr = dfutil(X_train1.iloc[:,1:10])
    mouth = dfutil(y_train1.to_frame())
    predictmodel = predict_model()
    predictmodel.fit(moneys,attr,mouth,3)
    odata = moneys
    oattr = attr
    omouth = mouth
    # rough_model = roughset(X_train1.iloc[:,1:10],y_train1)
    # a = rough_model.deal()
    # rough = a[0]
    filepath = './断案/0_0.09.csv'
    df = pd.read_csv(filepath)
    X_test1 = df.iloc[:, 2:12]
    moneys = dfutil(X_test1.iloc[:, 0].to_frame(),1)
    data = []
    for i in moneys:
        data.append([process_money(i[0], 1000)])
    attrs = dfutil(X_test1.iloc[:,1:10])
    moneys = dfutil(X_test.iloc[:, 0].to_frame(),1)
    data1 = []
    for i in moneys:
        data1.append([process_money(i[0], 1000)])
    attrs1 = dfutil(X_test.iloc[:,1:10])
    labels11 = dfutil(y_test.to_frame())
    abl_Time =5
    maelist=[]
    mselist = []
    percentlist = []
    perc_cntlist=[]
    # old_rough = title
    for i in range(abl_Time):
        new_money = []
        new_attrs =[]
        new_mouth = []
        numlist = []
        numlist1 = []
        attrsold = []
        attrsnew = []
        attrs1new = []
        for k in title:
            if k in rough:
                numlist.append(title.index(k))
        for k in title:
            if k in old_rough:
                numlist1.append(title.index(k))
        numlist.sort()
        numlist1.sort()
        for money,attr in zip(data,attrs):
            att = []
            for k in numlist1:
                att.append(attr[k])
            mouth_hat = predictmodel.predict([money],[att])
            fact = [money]
            for t in attr:
                fact.append(t)
            if abucemodel.abuce(fact,mouth_hat):
                new_money.append(money)
                new_attrs.append(attr)
                new_mouth.append(int(mouth_hat))
        for p in new_attrs:
            plist=[]
            for k in numlist:
                plist.append(p[k])
            attrsnew.append(plist)
        for p in attrs1:
            plist=[]
            for k in numlist:
                plist.append(p[k])
            attrs1new.append(plist)
        Flag = True
        for i in old_rough:
            if i not in rough:
                Flag = False
        if Flag:
            predictmodel.fit(new_money,attrsnew,new_mouth,3)
        else:
            attrsnew1 = []
            for p in oattr:
                plist = []
                for k in numlist:
                    plist.append(p[k])
                attrsnew1.append(plist)
            predictmodel.fit(odata, attrsnew1, omouth, 3)
            predictmodel.fit(new_money, attrsnew, new_mouth, 3)
        dfattrs = pd.DataFrame(new_attrs)
        dfmouth = pd.DataFrame(new_mouth)
        rough_model = roughset(dfattrs,dfmouth)
        a = rough_model.deal()
        old_rough=[]
        for k in rough:
            old_rough.append(k)
        rough = []
        for k in a[0]:
            rough.append(title[k])
        mae, mse, percent, perc_cnt=predictmodel.test(data1,attrs1new,labels11)
        maelist.append(mae)
        mselist.append(mse)
        percentlist.append(percent)
        perc_cntlist.append(perc_cnt)
    print(maelist)
    print(mselist)
    print(percentlist)
    print(perc_cntlist)