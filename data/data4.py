# -*- coding: utf-8 -*-
# @Time     :2019/10/13 18:39
# @Author   :XiaoMa
# @File     :data4.py
import pandas as pd
df=pd.read_csv('submit5.csv')
df.drop('Unnamed: 0',axis=1,inplace=True)

df.to_csv('submit5.csv',index=None)
