import numpy as np
# from roughtset import roughset
from collections import Counter
import itertools
import pandas as pd
class rule_tt():
    def __init__(self,xdata,ydata):
        self.xdata = xdata
        self.ydata = ydata

    def getlist(self,keylist,itemlist):

        traindata = self.xdata.loc[:, keylist[0:-1]]
        old_traindata = traindata
        for i in range(len(keylist) - 1):
            if '>' in str(itemlist[i]):
                traindata = traindata.loc[traindata[keylist[i]] > itemlist[i][1]]
                old_traindata = old_traindata.loc[old_traindata[keylist[i]]<=itemlist[i][1]]
            if '<' in str(itemlist[i]):
                traindata = traindata.loc[traindata[keylist[i]] < itemlist[i][1]]
                old_traindata = old_traindata.loc[old_traindata[keylist[i]]>=itemlist[i][1]]
            if '>=' in str(itemlist[i]):
                traindata = traindata.loc[traindata[keylist[i]] >= itemlist[i][1]]
                old_traindata = old_traindata.loc[old_traindata[keylist[i]]>itemlist[i][1]]
            if '<=' in str(itemlist[i]):
                traindata = traindata.loc[traindata[keylist[i]] <= itemlist[i][1]]
                old_traindata = old_traindata.loc[old_traindata[keylist[i]]<itemlist[i][1]]
            if '==' in str(itemlist[i]):
                traindata = traindata.loc[traindata[keylist[i]] == itemlist[i][1]]
                old_traindata = old_traindata.loc[old_traindata[keylist[i]]!=itemlist[i][1]]
            if '!=' in str(itemlist[i]):
                traindata = traindata.loc[traindata[keylist[i]] != itemlist[i][1]]
                old_traindata = old_traindata.loc[old_traindata[keylist[i]]==itemlist[i][1]]
        ndata = traindata
        indxlist = traindata.index.tolist()
        indxlist2 = old_traindata.index.tolist()
        nlsit = []
        olsit = []
        for i in range(len(indxlist)):
            if itemlist[-1][0]=='==':
                if self.ydata[indxlist[i]] == itemlist[-1][1]:
                    nlsit.append(indxlist[i])
                else:
                    olsit.append(indxlist[i])
                ylist = self.ydata[self.ydata==itemlist[-1][1]]
            else:
                if self.ydata[indxlist[i]] != itemlist[-1][1]:
                    nlsit.append(indxlist[i])
                else:
                    olsit.append(indxlist[i])
                ylist = self.ydata[self.ydata!=itemlist[-1][1]]
        return indxlist,indxlist2,nlsit,olsit,ylist

    def rule_reduction(self,rulelist,par=0.9):
        newrulelist = []
        rulelist = self.rule_delete(rulelist,par)
        for k in rulelist:
            k = self.rule_check(k)
            keylist1 = []
            itemlist = []
            for key,item in k.items():
                keylist1.append(key)
                itemlist.append(item)
            uplist, neglist, poslist, nposlist, ylist = self.getlist(keylist1,itemlist)
            if poslist!=[]:
                ifc = False
                a=1
                while a ==1:
                    cflag = False
                    i=0
                    while i < (len(keylist1)-1):
                        tempkey = []
                        tempitem = []
                        for j in range(len(keylist1)):
                            if j != i :
                                tempkey.append(keylist1[j])
                                tempitem.append(itemlist[j])
                        tup,tneg,tpos,tnpos,ty = self.getlist(tempkey,tempitem)
                        if uplist == tup:
                            keylist1 = tempkey
                            itemlist = tempitem
                            cflag = True
                            ifc = True
                            break
                        i = i +1
                    if cflag == False:
                        a=0
                if ifc ==True:
                    newrule = {}
                    for i in range(len(keylist1)):
                        newrule[keylist1[i]]=itemlist[i]
                    newrulelist.append(newrule)
                else:
                    newrulelist.append(k)
        return newrulelist

    def cercheck(self,rule):
        keylist = []
        itemlist = []
        for key, item in rule.items():
            keylist.append(key)
            itemlist.append(item)
        uplist, neglist, poslist, nposlist, ylist = self.getlist(keylist, itemlist)
        if len(uplist)!=0:
            cer = len(poslist)/len(uplist)
        else:
            cer = 0
        return cer

    def covcheck(self,rule):
        keylist = []
        itemlist = []
        for key, item in rule.items():
            keylist.append(key)
            itemlist.append(item)
        uplist, neglist, poslist, nposlist, ylist = self.getlist(keylist, itemlist)
        if len(ylist)!=0:
            cov = len(poslist)/len(ylist)
        else:
            cov = 0
        return cov

    def supcheck(self,rule):
        keylist = []
        itemlist = []
        for key, item in rule.items():
            keylist.append(key)
            itemlist.append(item)
        uplist, neglist, poslist, nposlist, ylist = self.getlist(keylist, itemlist)
        sup = len(poslist)/self.xdata.shape[0]
        return sup

    def rule_check(self,rule):
        keylist =self.xdata.columns.tolist()
        rulekey = list(rule.keys())
        rulekey = rulekey[0:-1]
        for key in rulekey:
            if key not in keylist:
                del rule[key]
        return rule

    def allcomb(self,l):
        num = 0
        for i in l:
            if num == 0:
                result = i
                num = 1
            else:
                result = list(itertools.product(result, i))
                tmp = []
                flag = 0
                for j in result:
                    if type(j[0]) == tuple:
                        a = list(j[0])
                        a.append(j[1])
                        tmp.append(a)
                        flag = 1
                    elif type(j[0]) == list:
                        a = []
                        for k in j[0]:
                            a.append(k)
                        a.append(j[1])
                        tmp.append(a)
                        flag = 1
                if flag == 1:
                    result = tmp
        return result

    def norule_gen(self,flag = 'all'):
        res = Counter(self.ydata)
        valuedic = {}
        for key,item in res.items():
           valuedic[key] = [idx for idx,i in enumerate(self.ydata) if i!=key]
        xvaluedic = {}
        xkey = []
        for i in self.xdata.columns.tolist():
            alist = self.xdata.loc[:,i]
            res = Counter(alist)
            xdic = {}
            keylist = []
            for key,item in res.items():
                keylist.append(key)
                xdic[key] = [idx for idx,i in enumerate(alist) if i!=key]
            xvaluedic[i] = xdic
            xkey.append(keylist)
        if flag == 'all':
            maprule = self.allcomb(xkey)
            rulelist = []
            maprulelist = []
            for i in maprule:
                rule = {}
                for j in range(len(i)):
                    rule[self.xdata.columns.tolist()[j]] = ['!=',i[j]]
                    if j == 0:
                        ruleset = set(xvaluedic[self.xdata.columns.tolist()[j]][i[j]])
                    else:
                        ruleset = set(xvaluedic[self.xdata.columns.tolist()[j]][i[j]]).intersection(ruleset)
                if len(list(ruleset))!=0:
                    maprulelist.append(list(ruleset))
                    rulelist.append(rule)
            nondic = {}
            bonddic = {}
            result1 = []
            result2 = []
            for key,item in valuedic.items():
                nondic[key]=[]
                bonddic[key]=[]
                for j in range(len(maprulelist)):
                    if set(maprulelist[j]).issubset(item):
                        nondic[key].append(j)
                        newrule = rulelist[j]
                        newrule['ans'] = ['!=',key]
                        result1.append(newrule)
                    elif len(list(set(maprulelist[j]).intersection(item)))!=0:
                        bonddic[key].append(j)
                        newrule = rulelist[j]
                        newrule['ans'] = ['!=', key]
                        result2.append([newrule,len(list(set(maprulelist[j]).intersection(item)))/len(maprulelist[j])])
        return result1,result2,nondic,bonddic,maprulelist

    def rule_gen(self,flag = 'all'):
        res = Counter(self.ydata)
        valuedic = {}
        for key,item in res.items():
           valuedic[key] = [idx for idx,i in enumerate(self.ydata) if i==key]
        xvaluedic = {}
        xkey = []
        for i in self.xdata.columns.tolist():
            alist = self.xdata.loc[:,i]
            res = Counter(alist)
            xdic = {}
            keylist = []
            for key,item in res.items():
                keylist.append(key)
                xdic[key] = [idx for idx,i in enumerate(alist) if i==key]
            xvaluedic[i] = xdic
            xkey.append(keylist)
        if flag == 'all':
            maprule = self.allcomb(xkey)
            rulelist = []
            maprulelist = []
            for i in maprule:
                rule = {}
                for j in range(len(i)):
                    rule[self.xdata.columns.tolist()[j]] = ['==',i[j]]
                    if j == 0:
                        ruleset = set(xvaluedic[self.xdata.columns.tolist()[j]][i[j]])
                    else:
                        ruleset = set(xvaluedic[self.xdata.columns.tolist()[j]][i[j]]).intersection(ruleset)
                if len(list(ruleset))!=0:
                    maprulelist.append(list(ruleset))
                    rulelist.append(rule)
            nondic = {}
            bonddic = {}
            result1 = []
            result2 = []
            for key,item in valuedic.items():
                nondic[key]=[]
                bonddic[key]=[]
                for j in range(len(maprulelist)):
                    p = maprulelist[j]
                    q = item
                    if set(maprulelist[j]).issubset(item):
                        nondic[key].append(j)
                        newrule = rulelist[j]
                        newrule['ans'] = ['==',key]
                        result1.append(newrule)
                    elif len(list(set(maprulelist[j]).intersection(item)))!=0:
                        bonddic[key].append(j)
                        newrule = rulelist[j]
                        newrule['ans'] = ['==', key]
                        result2.append([newrule,len(list(set(maprulelist[j]).intersection(item)))/len(maprulelist[j])])
        return result1,result2,nondic,bonddic,maprulelist

    def rule_merge(self,rulist):
        ruledic={}
        for i in range(len(rulist)):
            rule = rulist[i]
            if rule['ans'][1] not in list(ruledic.keys()):
                ruledic[rule['ans'][1]]=[]
                ruledic[rule['ans'][1]].append(rule)
            else:
                ruledic[rule['ans'][1]].append(rule)
        xlable = self.xdata.columns.tolist()
        rule_merge = []
        mergerule = []
        for i in list(ruledic.keys()):
            tmprule={}
            rulist = ruledic[i]
            if len(rulist)!=1:
                setlist = []
                a = rulist[0]
                item = a[list(a.keys())[0]][0]
                for j in rulist:
                    tmplist = ['nan']*len(xlable)
                    b=list(j.keys())[0:-1]
                    for key in list(j.keys())[0:-1]:
                        idx = xlable.index(key)
                        tmplist[idx]=j[key][1]
                    setlist.append(tmplist)
                dfset = pd.DataFrame(setlist)
                dfset.columns = xlable
                flag = 0
                for j in xlable:
                    anslist = dfset.loc[:,j]
                    res = Counter(anslist)
                    a = list(res.keys())
                    if len(a)==1 and a[0]!='nan':
                        tmprule[j] = [item,a[0]]
                        flag =1
                if flag !=1:
                    num = 0
                    df1set = dfset
                    tmpset = dfset
                    while num!=len(df1set.index.tolist()):
                        dfset2 = tmpset
                        tmprule={}
                        N=0
                        for j in xlable:
                            anslist = dfset2.loc[:, j]
                            res = Counter(anslist)
                            a = list(res.keys())
                            if len(a)==2 and 'nan' in a:
                                for q1 in a:
                                    if q1!='nan':
                                        p=q1
                                xlist = [idx for idx, q in enumerate(anslist) if q == p]
                                num = num +len(xlist)
                                tmplist = list(set(dfset2.index.tolist())-set(xlist))
                                tmpset = tmpset.iloc[tmplist,:].reset_index(drop=True)
                                dfset2 = dfset2.iloc[xlist,:].reset_index(drop=True)
                                N=1
                                for k in xlable:
                                    anslist = dfset2.loc[:,k]
                                    res = Counter(anslist)
                                    a = list(res.keys())
                                    if len(a) == 1 and a[0] != 'nan':
                                        tmprule[j] = [item, a[0]]
                                break
                            elif len(a)==1 and a[0]!='nan':
                                tmprule[j] = [item, a[0]]
                                num = len(df1set.index.tolist())
                                N=1
                                # tmprule[j] = [item]
                                # for anskey in a:
                                #     if anskey!='nan':
                                #         tmprule[j].append(anskey)
                            # else:
                            #     if len(a) == 1 and a[0]!='nan':
                            #         tmprule[j] = [item, a[0]]
                        if N==1:
                            tmprule['ans'] = [item, i]
                            rule_merge.append(tmprule)
                        else:
                            for j in dfset2.index.tolist():
                                rule = {}
                                for k in xlable:
                                    if dfset2.loc[j,k]!='nan':
                                        rule[k] = [item,dfset2.loc[j,k]]
                                rule['ans']=[item,i]
                                rule_merge.append(rule)
                            num = len(df1set.index.tolist())
                else:
                    tmprule['ans']=[item,i]
                    rule_merge.append(tmprule)
            else:
                rule_merge.append(rulist[0])
            min = len(list(rule_merge[0].keys()))
            tmpmerge = []
            for k in rule_merge:
                if len(list(k.keys()))==min:
                    tmpmerge.append(k)
                elif len(list(k.keys()))<min:
                    tmpmerge = []
                    tmpmerge.append(k)
                    min = len(list(k.keys()))
            rule_merge = []
            for k in tmpmerge:
                mergerule.append(k)

        return mergerule


        # mergerule = {}
        # for i in list(ruledic.keys()):
        #     rulelist = ruledic[i]
        #     flag = 0
        #     if i not in list(mergerule.keys()):
        #         mergerule[i]=[]
        #     while flag < len(rulelist):
        #         delrule = []
        #         tmprule = {}
        #         num = 0
        #         for j in list(rulelist[0].keys()):
        #             if delrule==[]:
        #                 for k in rulelist:
        #                     if j in list(k.keys()) and rulelist[0][j]==k[j]:
        #                         delrule.append(k)
        #                         num = num+1
        #                         flag = flag+1
        #                 if num==len(rulelist):
        #                     tmprule[j]=rulelist[0][j]
        #             elif len(delrule)==len(rulelist):
        #                 delrule1 = []
        #                 for k in rulelist:
        #                     if j in list(k.keys()) and rulelist[0][j]==k[j]:
        #                         delrule1.append(k)
        #                         num = num+1
        #                         flag = flag+1
        #                 if num==len(rulelist):
        #                     tmprule[j]=rulelist[0][j]
        #             else:
        #                 num = 0
        #                 for k in delrule:
        #                     if j in list(k.keys()) and rulelist[0][j]==k[j]:
        #                         num = num+1
        #                 if num==len(delrule):
        #                     tmprule[j]=rulelist[0][j]
        #     mergerule[i].append(tmprule)

    def rule_delete(self,rulist,par):
        newrulist=[]
        for k in rulist:
            a = self.cercheck(k)
            if a >par:
                newrulist.append(k)
        return newrulist




if __name__=='__main__':
    import pandas as pd
    data = pd.read_csv('./断案/1_0.90.csv')
    xdata = data.loc[:,['sum','attitude_bool']]
    oldxdata = xdata
    xdata = xdata.loc[xdata['sum']>100]
    print(oldxdata)
    print(xdata.columns.tolist())
    ndata = xdata
    print(xdata)
    oldxdata = pd.DataFrame(oldxdata.loc[1,:]).T
    print(oldxdata)
    print(type(oldxdata))
    print(oldxdata.columns.tolist())
