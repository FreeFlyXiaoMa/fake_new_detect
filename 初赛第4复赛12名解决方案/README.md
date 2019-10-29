# FalsenewsDetection

### 个人比赛总结

智源&计算所虚假新闻挑战赛的task1 [**False News Text Detection**](https://www.biendata.com/competition/falsenews/) **初赛成绩 top-4**，**复赛成绩 top-12**

模型无特别之处，直接用的LM base (roBERTa 和 XLNet)，科大讯飞训练的版本

用large的batchsize太小效果不如base，机器好的请忽略

在保证截断长度能覆盖大部分数据的情况下，batchsize越大越好，这里截断长度192，batchsize24，差不多11G显存跑满

大部分模型evaluate的情况都是FN比FP要高几倍，投票策略单纯五五开的话效果不理想(10折单模初赛线上只有0.89左右)

所以投票策略应该侧重于提高正类的召回

用投票完的结果作为脏标签最后再过一遍训练，结果能辅助稳定提升3-4个千分点(无论在初赛还是复赛)

复赛个人试了stacking效果不行，坐等复赛前排大佬开源分享

### 线上分数

**初赛:** 

- 10fold bert-wwm-base + 10fold xlnet-base                     (投票策略)            线上0.92977

- 10fold bert-wwm-base + 10fold xlnet-base + 15fold xlnet-base (投票策略和脏标签辅助)  线上0.93337

**复赛:** 

- 13fold roBERTa-base + 10fold xlnet-base + 15fold xlnet-base  (投票策略和脏标签辅助)  线上0.89012

### 代码环境

在**ubuntu**下使用的 [**Deepo**](https://github.com/ufoym/deepo) **docker**服务，容器版本 **tensorflow-py36-cu90**

|环境/库|版本|
|:---------:|----------|
|**ubuntu**|16.04 LTS|
|**python**|3.6.8|
|**tensorflow**|1.12.0|

### 代码详情

请着重关注 **run_kfold_falsenews.sh** 这个脚本, 负责数据集分割和进行交叉验证

详细的训练参数可在 **run_classifer.py** 中设置 或者 **run_kfold_faslsenews.sh** 中指定 (保留了比赛训练参数) 

其中 **do_train_test** 和 **train_test_set** 两个参数用来训练脏标签
