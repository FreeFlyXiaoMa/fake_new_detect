# -*- coding: utf-8 -*-
# @Time     :2019/10/9 13:13
# @Author   :XiaoMa
# @File     :data3.py

import pandas as pd
data=pd.read_csv('duplicate_remove_debunking.csv')

#训练集总结，有将消息传播的嫌疑;含有夸张成分词;含有犯罪的词；社会负面
# rumors=['再也不敢','求扩散','转发出去','让更多的人看到','求转','转发','一定要转','速度转发','群发扩散','爱心接力',
#         '转贴','拐卖','杀戮','出现意外','帮忙转','太恐怖','帮忙扩散','都传下','这是事实','以后千万','可不转','自然会转','砍','死','死亡',
#         '请告诉','求快速转发','求转发','以后再也','以后不要','一次性筷子','太可怕',
#         '一路走好','急疯了','有知情者','都转下','大家都注意','尋人啟事','寻人启事','大爆炸',
#         '猫狗肉','转出去','转发下','Shit','Tmd','求转','帮朋友转','闯入','帮忙转发','帮忙扩散','就转',
#         '转出去','帮忙转','帮朋友转','免流量','尼姑','还会玩吗','假鸡蛋','已证实','棉花','塑料','诺基亚','轮奸'
#         ]

rumors=['再也不敢','求扩散','转发出去','让更多的人看到','求转','转发','一定要转','速度转发','群发扩散','爱心接力',
        '帮忙转','太恐怖','帮忙扩散','都传下','可不转','自然会转',#'砍','死','死亡',
        '请告诉','求快速转发','求转发','以后再也','以后不要','一次性筷子','太可怕',
        '一路走好','急疯了','有知情者','都转下','大家都注意','尋人啟事','寻人启事','大爆炸',
        '猫狗肉','转出去','转发下','求转','帮朋友转','帮忙转发','帮忙扩散','就转',
        '转出去','帮忙转','帮朋友转','假鸡蛋','已证实','棉花','塑料','诺基亚','轮奸'
        ]

df_test=pd.read_csv('test_stage1.csv')
df_pred=pd.read_csv('baseline2.csv')

#对预测结果做矫正
result_list=[]
for index in df_test.index:
        label = 9999
        text=df_test['text'][index]
        for item in rumors:
                if item in text:
                        label=1
        if label >1:
                result_list.append(0)
        else:
                result_list.append(1)

#将result_list中的假新闻更正到预测列表中
count=0
for index in df_pred.index:
        if result_list[index]==1:
                re=df_pred['label'][index]
                if re==0:
                        df_pred['label'][index]=1
                        count+=1
# df_pred['pre']=result_list
# df_pred['text']=df_test['text']

# df_pred.to_csv('df_pred.csv',index=None)
print('count:',count)
df_pred.to_csv('new.csv',index=None)


