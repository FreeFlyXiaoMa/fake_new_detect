# -*- coding: utf-8 -*-
# @Time     :2019/9/26 9:51
# @Author   :XiaoMa
# @Site     :
# @File     :eda.py
# @Software :PyCharm

import pandas as pd

from sklearn.model_selection import train_test_split

train=pd.read_csv('./data/train.csv')
label=train['label']
text=train['text']

with open('./data/val.txt','w') as f:
    for i in range(5000):
        label_i=label[i]
        text_i=text[i]
        f.write(str(label_i)+ '\t' +text_i +'\n')

with open('./data/train.txt','w') as file:
    for j in range(5000,len(label)):
        label_j=label[j]
        text_j=text[j]
        file.write(str(label_j) + '\t' +text_j + '\n')
