import numpy as np
from collections import Counter
class abuductionmodel():
    def __init__(self,map=None):
        self.reasondict = {'cat':1,'nocat':0}
        self.map = map

    def abuce(self,fact):
        for i in range(fact.shape[0]):
            if fact[i,0]==0:
                if fact[i,1]!=1:
                    fact[i,1]=1
            else:
                if fact[i,1]!=0:
                    fact[i,1]=0
        return fact
    def abuce1(self,fact,predict):
        dellist=[]
        for i in range(fact.shape[0]):
            if fact[i,0]==0:
                if fact[i,1]<2:
                    a = predict[i, :].tolist()
                    b = sorted(a, reverse=True)
                    # b.sort(reverse=True)
                    if a.index(b[1]) >1:
                        if b[1] > 0.9:
                            fact[i, 1] = a.index(b[1])
                        else:
                            dellist.append(i)
                    else:
                        dellist.append(i)
            else:
                if fact[i,1]>1:
                    a=predict[i,:].tolist()
                    b=sorted(a,reverse=True)
                    #b.sort(reverse=True)
                    if a.index(b[1])<2:
                        if b[1]>0.9:
                            fact[i,1]=a.index(b[1])
                        else:
                            dellist.append(i)
                    else:
                        dellist.append(i)
        return fact,dellist
    def abuce2(self,fact,predict):
        dellist=[]
        for i in range(fact.shape[0]):
            if fact[i,0]==0:
                if fact[i,1]!=1:
                    fact[i,1]=1
            elif fact[i,0]==2:
                if fact[i,1]!=0:
                    fact[i,1]=0
            else:
                if fact[i,1]==1 or fact[i,1]==0:
                    a = predict[i, :].tolist()
                    b = sorted(a, reverse=True)
                    # b.sort(reverse=True)
                    if a.index(b[1]) >1:
                        if b[1] > 0.9:
                            fact[i, 1] = a.index(b[1])
                        else:
                            dellist.append(i)
                    else:
                        dellist.append(i)
        return fact,dellist

    def abucemain(self,fact,predict):
        dellist=[]
        for i in range(fact.shape[0]):
            for key,value in self.map.items():
                if int(key)==fact[i,0]:
                    if fact[i,1] not in value:
                        a = predict[i, :].tolist()
                        b = sorted(a, reverse=True)
                        # b.sort(reverse=True)
                        if a.index(b[1]) in value:
                            if b[1] > 0.9:
                                fact[i, 1] = a.index(b[1])
                            else:
                                dellist.append(i)
                        else:
                            dellist.append(i)
        return fact,dellist

    # def abuce(self,fact,predict,type,type2):
    #     dellist=[]
    #     clist=[]
    #     countlist={}
    #     maxlist = []
    #     for i in range(type):
    #         a = np.where(fact[:,0]==i)[0]
    #         clist.append(a)
    #     num=0
    #     for j in clist:
    #         b = list(fact[j,1])
    #         countlist[num] = []
    #         for k in range(type2):
    #             countlist[num].append(b.count(k))
    #         number = Counter(countlist[num])
    #         result = number.most_common()
    #         maxlist.append(result[0][0])
    #     if len(maxlist)==len(set(maxlist)):
    #         typelist = []
    #         for i in maxlist:
    #             for key,items in self.map.items():
    #                 if i in items:
    #                     typelist.append(key)
    #                     continue
    #         if len(typelist)==len(set(typelist)):
    #             newmap={}
    #             for i in range(typelist):
    #                 if len(self.map[typelist])>1:
    #
    #         else:
    #             fact, dellist = self.abucemain(fact, predict)
    #             map = self.map
    #     else:
    #                         fact,dellist = self.abucemain(fact,predict)
    #         map = self.map
    #     return fact,dellist,map

"""
  def gen_map(self,chars,symbs):
        n_char = len(chars)
        n_symbs = len(symbs)
        if n_char != n_symbs:
            print('Characters and symbols size dosen\'t match.')
            return
        from itertools import permutations
        mappings = []
        # returned mappings
        perms = permutations(symbs)
        for p in perms:
            mappings.append(dict(zip(chars, list(p))))
        return mappings

    def testmap(self,mappings,exs,no_change):
        for m in mappings:


"""

