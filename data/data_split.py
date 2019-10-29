# -*- coding: utf-8 -*-
# @Time     :2019/10/13 9:54
# @Author   :XiaoMa
# @File     :data_split.py

import pandas as pd

#重新分配训练集和验证集
#验证集由原来的5000样本修改为2000
train=pd.read_csv('train.csv')  #38471
train_36471=train.loc[0:36471]
valid_2000=train.loc[36271:38471]


train_36471.to_csv('train_36471.csv',index=None)
valid_2000.to_csv('valid_2000.csv',index=None,)

file=open('valid_2000.txt','w')
with open('valid_2000.csv','r') as f:
    lines=f.readlines()
    for line in lines[1:-1]:

        re=line.strip().split(',')
        id=re[0]
        text=re[1]
        label=re[2]
        file.write(label+'\t' +text+'\n')
file.close()

file2=open('train_36471.txt','w')
with open('train_36471.csv','r') as f:
    lines=f.readlines()
    for index,line in enumerate(lines[1:-1]):

        re=line.strip().split(',')
        id=re[0]
        text=re[1].replace(r'\n','')
        label=re[2]
        file2.write(label+'\t'+text+'\n')

file2.close()





