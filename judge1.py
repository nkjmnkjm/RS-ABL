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
    # rulepath = './断案/rule_file.txt'
    # abucemodel = abuce_model(rulepath)
    # filepath = './断案/1_0.90.csv'
    filepath = './断案/0_0.10.csv'
    df = pd.read_csv(filepath)
    from sklearn import model_selection

    X_train = df.iloc[:, 2:12]
    y_train = df.iloc[:, 12]
    ruledict = {'top': [], 'mid': [], 'bot': []}
    # X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)
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
    # print('*************')
    # print(mae)
    # print(mse)
    # print(percent)
    # print(perc_cnt)
    # print('************')
    # X_train1, X_test1, y_train1, y_test1 = model_selection.train_test_split(X_train, y_train, test_size=0.9)
    X_train1 = X_train
    y_train1 = y_train

    moneys = dfutil(X_train1.iloc[:, 0].to_frame(), 1)
    moneyslist = []
    for i in moneys:
        moneyslist.append([process_money(i[0], 1000)])
    new_moneys1 = moneyslist
    title = X_train1.iloc[:, 1:10].columns.values.tolist()
    new_attr1 = dfutil(X_train1.iloc[:, 1:10])
    new_mouth1 = dfutil(y_train1.to_frame())
    predictmodel = predict_model()
    predictmodel.fit(new_moneys1, new_attr1, new_mouth1, 3)
    mae, mse, percent, perc_cnt = predictmodel.test(data1, attrs1, labels11)
    X_train1.iloc[:,0]=new_moneys1
    rates=[]
    for money1,month in zip(new_moneys1,new_mouth1):
        baseline = predictmodel.predict_baseline([money1])
        rates.append(int(np.round(((month - baseline) / baseline)*100,2)))
    rule_treat = rule_tt(X_train1.iloc[:,1:10],rates)
    result1, result2, nondic, bonddic, maprulelist = rule_treat.rule_gen()
    result1 = rule_treat.rule_reduction(result1)
    result1 = rule_treat.rule_merge(result1)
    for i in result1:
        ruledict['mid'].append(i)
    result1, result2, nondic, bonddic, maprulelist = rule_treat.norule_gen()
    # rulelen.append(a+len(result1))
    result1 = rule_treat.rule_reduction(result1)
    result1 = rule_treat.rule_merge(result1)
    for i in result1:
        ruledict['mid'].append(i)
    from abuce_judge import new_abuce
    abucemodel = new_abuce(ruledict)
    odata = moneys
    oattr = attr
    omouth = mouth
    # rough_model = roughset(X_train1.iloc[:,1:10],y_train1)
    # a = rough_model.deal()
    # rough = a[0]
    filepath = './断案/1_0.90.csv'
    df = pd.read_csv(filepath)
    X_test1 = df.iloc[:, 2:12]
    moneys = dfutil(X_test1.iloc[:, 0].to_frame(), 1)
    data = []
    for i in moneys:
        data.append([process_money(i[0], 1000)])
    X_test1.iloc[:,0]=data
    attrs = dfutil(X_test1.iloc[:, 1:10])
    # moneys = dfutil(X_test.iloc[:, 0].to_frame(), 1)
    # data1 = []
    # for i in moneys:
    #     data1.append([process_money(i[0], 1000)])
    # attrs1 = dfutil(X_test.iloc[:, 1:10])
    # labels11 = dfutil(y_test.to_frame())
    abl_Time = 1
    maelist = []
    mselist = []
    percentlist = []
    perc_cntlist = []
    for i in range(abl_Time):
        moneys = dfutil(X_train1.iloc[:, 0].to_frame(), 1)
        moneyslist = []
        for i in moneys:
            moneyslist.append([process_money(i[0], 1000)])
        new_money = moneyslist
        title = X_train1.iloc[:, 1:10].columns.values.tolist()
        new_attrs = dfutil(X_train1.iloc[:, 1:10])
        new_mouth = dfutil(y_train1.to_frame())
        mouth_hat = predictmodel.predict(data,attrs)
        mouth_hat = list(mouth_hat)
        mouth = []
        for k in mouth_hat:
            mouth.append(round(k))
        mouth_hat = np.array(mouth)
        # rates = []
        # for money1, month in zip(data, mouth_hat):
        #     baseline = predictmodel.predict_baseline(money1)
        #     rates.append(round((month - baseline) / baseline), 2)
        rates = predictmodel.predict_rate(attrs)
        rates = list(np.round(rates*100,2))
        new_fact, dellist, ruledict = abucemodel.abuce(X_test1.iloc[:, 1:10], rates)
        tmpey = new_fact
        tmpex = X_test1
        a = list(set(range(len(tmpey))) - set(dellist))
        tmpex = tmpex.iloc[a, :].reset_index(drop=True)
        alist = []
        for idx in a:
            b=[X_test1.iloc[idx,0]]
            baseline = predictmodel.predict_baseline([b])
            t = np.multiply(baseline,tmpey[idx]/100+1)
            alist.append(t)
        tmpey = alist
        if len(a) != 0:
            for j in range(0,len(a)):
                new_money.append([tmpex.iloc[j,0].tolist()])
                b=[]
                for k in tmpex.iloc[j,1:10].tolist():
                    b.append(int(k))
                new_attrs.append(b)
                new_mouth.append(int(tmpey[j]))
        predictmodel.fit(new_money, new_attrs, new_mouth, 3)
    mae, mse, percent, perc_cnt = predictmodel.test(data1, attrs1, labels11)
    print('*************')
    print(mae)
    print(mse)
    print(percent)
    print(perc_cnt)
    print('************')