## 一. 环境搭建  

建议为python3.6，

同时需要cuda10.1和配套的cudnn7.6.5，

具体需要安装的包可参见requirements。

## 二. 训练模型

可参见main.py

如果需要重新finetune，需要调整`LongTxtBertMultiLearner.fit`参数为`True`

否则，则设置为`False`

## 三. 预测

可参见main.py

## 四. 其他设置

setting.conf主要为数据库账号信息，

LocalEnv.py主要为路径设置和一些模型相关的设置。