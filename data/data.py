# -*- coding: utf-8 -*-
# @Time     :2019/9/30 13:08
# @Author   :XiaoMa
# @Site     :
# @File     :data.py
# @Software :PyCharm

import pandas as pd
pd.set_option('display.max_columns',1000)
pd.set_option('display.width',1000)
pd.set_option('display.max_rows',1000)

test=pd.read_csv('test_stage1.csv')

train=pd.read_csv('train.csv')
text_list=[]
label_list=[]

for index in train.index:

    if 256 < len(train['text'][index]) <=512 :
        text=train['text'][index]
        label=train['label'][index]
        text_a=text[:255]
        text_b=text[255:len(train['text'][index])]

        text_list.append(text_a)
        label_list.append(label)

        if len(text_b) > 35:
            text_list.append(text_b)
            label_list.append(label)
        train.drop(index=index,axis=0,inplace=True)
        # print(len(label_list))

train.reset_index(inplace=True)
train.drop('index',axis=1,inplace=True)

for index in train.index:
    if 512 < len(train['text'][index]) <=1024:
        text=train['text'][index]
        label=train['label'][index]

        text_a=text[:255]
        text_b=text[255:512]
        text_c=text[512:len(train['text'][index])]

        if 35 < len(text_c) <=256:      #text_c 长度介于512~768
            text_list.append(text_c)
            label_list.append(label)
        elif 256 < len(text_c) < 512:   #text_c 长度介于768~1024
            a=text_c[0:255]
            c=text_c[255:len(text_c)]

            text_list.append(a)
            label_list.append(label)

            if len(c) >35:
                text_list.append(c)
                label_list.append(label)
        train.drop(index=index,axis=0,inplace=True)
        # print(len(train))

train.reset_index(inplace=True)
train.drop('index',axis=1,inplace=True)

for index in train.index:
    if  1024 < len(train['text'][index]) <= 1536:
        text=train['text'][index]
        label=train['label'][index]

        text_a=text[0:255]
        text_b=text[255:512]
        text_c=text[512:768]
        text_d=text[768:1024]
        text_e=text[1024:len(train['text'][index])]

        text_list.append(text_a)
        label_list.append(label)

        text_list.append(text_b)
        label_list.append(label)

        text_list.append(text_c)
        label_list.append(label)

        text_list.append(text_d)
        label_list.append(label)

        if 35 < len(text_e) <= 256:
            text_list.append(text_e)
            label_list.append(label)
        elif 256 < len(text_e) <512:
            a=text_e[0:255]
            b=text_e[255:len(text_e)]

            text_list.append(a)
            label_list.append(label)

            if len(b)>35:
                text_list.append(b)
                label_list.append(label)
        train.drop(index=index,axis=0,inplace=True)

train.reset_index(inplace=True)
train.drop('index',axis=1,inplace=True)

for index in train.index:
    if 1536 < len(train['text'][index]) <=2048:
        text=train['text'][index]
        label=train['label'][index]
        text_a=text[0:255]
        text_b=text[255:512]
        text_c=text[512:768]
        text_d=text[768:1024]
        text_e=text[1024:1280]
        text_f=text[1280:1536]
        text_g=text[1536:len(train['text'][index])]

        text_list.append(text_a)
        label_list.append(label)

        text_list.append(text_b)
        label_list.append(label)

        text_list.append(text_c)
        label_list.append(label)

        text_list.append(text_d)
        label_list.append(label)

        text_list.append(text_e)
        label_list.append(label)

        text_list.append(text_f)
        label_list.append(label)

        if 35 < len(text_g) <= 256:
            text_list.append(text_g)
            label_list.append(label)

        elif len(text_g) > 256:
            a=text_g[0:255]
            b=text_g[255:len(text_g)]

            text_list.append(a)
            label_list.append(label)

            if len(b) >35:
                text_list.append(b)
                label_list.append(label)

        train.drop(index=index,axis=1,inplace=True)

train.reset_index(inplace=True)
train.drop('index',axis=1,inplace=True)

df_train=pd.DataFrame()

df_train['text']=text_list
df_train['label']=label_list
train.drop('id',axis=1,inplace=True)

augment_train=pd.concat([train,df_train],axis=0)
augment_train.to_csv('augment_train.csv')






