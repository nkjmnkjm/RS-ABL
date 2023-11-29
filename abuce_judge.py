import copy
import numpy as np
import pandas as pd
from collections import Counter
import random
def parse_condition(rule):
    ret = []
    for r in rule.split("^"):
        d = r.split(':')
        ret.append((int(d[0]), int(d[1])))
    return ret

class abuce_model():

    def __init__(self,filename,numlist=[]):
        if numlist==[] or len(numlist)!=3:
            self.LARGE = 1000
            self.HUGE = 30000
            self.EXTRA_HUGE = 300000
        else:
            self.LARGE = numlist[0]
            self.HUGE = numlist[1]
            self.EXTRA_HUGE = numlist[2]
        self.rule_list = []
        if filename is None:
            return
        with open(filename) as fin:
            for rule in fin:
                rule = rule.strip().replace(" ", "").split('#')
                left, right, p = rule
                left = parse_condition(left)
                right = parse_condition(right)
                self.rule_list.append((left, right, float(p)))

    def get_penalty_type(self,money,attrs):
        [no_damage, attitude, surrender, again, young, forgive, tool, room, theft] = attrs
        if money < self.LARGE:
            if room == 1 or theft == 1:
                return 0
            if money >= 500 and again == 1:
                return 0
        elif int(money) < self.HUGE:
            return 0
        elif int(money) < self.EXTRA_HUGE:
            return 1
        else:
            return 2
        return -1

    def abuce(self,fact,month):
        pro = self.judge(fact)

        money =fact[0][0]

        [no_damage, attitude, surrender, again, young, forgive, tool, room, theft] = fact[1:len(fact)]
        if pro < 1e-6:
            return False
        penalty_type = self.get_penalty_type(fact[0][0], fact[1:len(fact)])
        if penalty_type == -1:
            return False
        if penalty_type == 0:
            if month <= 3 * 12:
                return True
            elif month >= 3 * 12 and month <= 10 * 12 and room == 1 and int(money) >= 15000:
                return True
            else:
                return False
        if penalty_type == 1:
            if month >= 3 * 12 and month <= 10 * 12:
                return True
            if month <= 3 * 12 and surrender == 1:
                return True
            if month >= 10 * 12 and room == 1 and int(money) >= 150000:
                return True
        if penalty_type == 2:
            if month >= 10 * 12:
                return True
            if month <= 10 * 12 and month >= 3 * 12 and (surrender == 1 or young == 1):
                return True
        return False

    def __satisfied(self, status, rule):
        for r in rule:
            if status[r[0]] != r[1]:
                return False
        return True

    def fix(self, status):
        ret = []
        for rule in self.rule_list:
            left, right, p = rule

            if (not self.__satisfied(status, left)):
                continue
            new_status = copy.deepcopy(status)
            for r in right:
                new_status[r[0]] = r[1]
            ret.append((new_status, p))
        return ret

    def judge(self, status):
        total_p = 1
        for rule in self.rule_list:
            left, right, p = rule
            if self.__satisfied(status, left):
                if self.__satisfied(status, right):
                    total_p *= p
                else:
                    total_p *= (1 - p)
            if total_p < 1e-6:
                return total_p
        return total_p

    def abuce_test(self,fact,predict):
        dellist = []
        relist = []
        for i in range(fact.shape[0]):
            result = self.abuce(fact[i,:],predict[i,:])
            if result:
                relist.append(i)
            else:
                dellist.append(i)
        return relist

class new_abuce():
    def __init__(self,ruledic):
        self.toprulelist = ruledic['top']
        self.midrulelist = ruledic['mid']
        self.botrulelist = ruledic['bot']

    def satifiy(self,data,rule):
        for key,item in rule.items():
            if key!='ans':
                if key in data.columns.tolist():
                    if item[0]=='==':
                        if data.loc[0,key]!=item[1]:
                            return False
                    elif item[0]=='!=':
                        if data.loc[0,key]==item[1]:
                            return False
                    elif item[0]=='>=':
                        if data.loc[0,key]<item[1]:
                            return False
                    elif item[0]=='<=':
                        if data.loc[0,key]>item[1]:
                            return False
                    elif item[0]=='<':
                        if data.loc[0,key]>=item[1]:
                            return False
                    elif item[0]=='>':
                        if data.loc[0,key]<=item[1]:
                            return False
        return True

    def abuce(self,data,fact):
        # res  = Counter(fact)
        # keylist = []
        # for key,item in res.items():
        #     keylist.append(key)
        # for i in range(len(data.index.tolist())):
        #     for j in self.toprulelist:
        #         if self.satifiy(pd.DataFrame(data.loc[i,:]).T,j)==True:
        #             if j['ans'][0] == '=':
        #                 if fact[i] != j['ans'][1]:
        #                     fact[i] = j['ans'][1]
        #             elif j['ans'][0] == '!=':
        #                 if fact[i] == j['ans'][1]:
        #                     fact[i] = 'nan'
        dellist = []
        changelist = []
        Change = []
        for i in range(data.shape[0]):
            for j in self.toprulelist:
                if self.satifiy(pd.DataFrame(data.loc[i,:]).T.reset_index(drop=True),j):
                    if j['ans'][0]=='==':
                        if fact[i]!=j['ans'][1]:
                            fact[i] = j['ans'][1]
                            changelist.append(i)
                            Change.append(i)
                    elif j['ans'][0]=='!=':
                        if fact[i] == j['ans'][1]:
                            if i not in dellist:
                                dellist.append(i)
            for j in self.midrulelist:
                delrule = []
                if self.satifiy(pd.DataFrame(data.loc[i,:]).T.reset_index(drop=True),j):
                    if j['ans'][0]=='==':
                        if fact[i]!=j['ans'][1]:
                            if i not in changelist:
                                fact[i] = j['ans'][1]
                                changelist.append(i)
                            else:
                                if i not in Change:
                                    if i not in dellist:
                                        dellist.append(i)
                                delrule.append(j)
                    elif j['ans'][0]=='!=':
                        if fact[i] == j['ans'][1]:
                            if i not in changelist:
                                if i not in dellist:
                                    dellist.append(i)
                            else:
                                if i not in Change:
                                    if i not in dellist:
                                        dellist.append(i)
                                delrule.append(j)

            for k in delrule:
                self.botrulelist.append(k)
                self.midrulelist.remove(k)
            if len(delrule)!=0:
                print(len(delrule))
        newrule = {'top':self.toprulelist,'mid':self.midrulelist,'bot':self.botrulelist}
        return fact,dellist,newrule




