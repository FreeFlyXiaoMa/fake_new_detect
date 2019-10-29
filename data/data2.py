# -*- coding: utf-8 -*-
# @Time     :2019/10/9 12:12
# @Author   :XiaoMa
# @File     :data2.py

import pandas as pd

#尝试对debunking.csv对新闻去重
#debunking.csv共有辟谣文本37877条

debunking=pd.read_csv('debunking.csv')
id_set=set()
for index in debunking.index:
    if debunking['id'][index] not in id_set:
        id_set.add(debunking['id'][index])
    else:
        debunking.drop(index=index,axis=0,inplace=True)

# print(len(debunking))  去重后有33360条
debunking=debunking.reset_index()
debunking.drop(['index','Unnamed: 0'],axis=1,inplace=True)

debunking.to_csv('duplicate_remove_debunking.csv')

# 根据新闻内容去重
duplicate_remove_debunking=pd.read_csv('duplicate_remove_debunking.csv')
text_set=set()

for index in duplicate_remove_debunking.index:
    if duplicate_remove_debunking['text'][index] not in text_set:
        text_set.add(duplicate_remove_debunking['text'][index])
    else:
        duplicate_remove_debunking.drop(index=index,axis=0,inplace=True)

print(len(duplicate_remove_debunking))

duplicate_remove_debunking=duplicate_remove_debunking.reset_index()
duplicate_remove_debunking.drop(['index','Unnamed: 0'],axis=1,inplace=True)
duplicate_remove_debunking.to_csv('duplicate_remove_debunking.csv')




