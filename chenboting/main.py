#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/25 下午9:17
# @Author : Boting Chen
# @File : main.py
if __name__ == '__main__':
    from source.LocalEnv import INPUT_PATH, OUTPUT_PATH
    from source.ModelTrain.learner import pd, LongTxtBertMultiLearner, os, LABEL_COLS, path_base_bert, path_saved_bert

    # _goal = 'train'
    _goal = 'predict'
    _retrain = False

    if _goal == 'train':  # 如果训练
        train_ = pd.read_csv(os.path.join(INPUT_PATH, 'train.csv'))
        if _retrain:  # 如果重新finetune，模型从base bert读取
            learner_ = LongTxtBertMultiLearner.from_pretrained_model(train_, path_base_bert)
        else:  # 如果不重新finetune，则直接调用saved bert
            learner_ = LongTxtBertMultiLearner.from_pretrained_model(train_, path_saved_bert)
        learner_.fit(re_finetune=_retrain)
    else:  # 如果预测
        test_ = pd.read_csv(os.path.join(INPUT_PATH, 'test.csv')).iloc[:100, :].reset_index(drop=True)
        learner_ = LongTxtBertMultiLearner.from_pretrained_model(test_, path_saved_bert)
        res = pd.DataFrame(learner_.predict())
        res.columns = LABEL_COLS
        res = pd.concat([test_.iloc[:, :1], res], axis=1)
        res.to_csv(os.path.join(OUTPUT_PATH, 'res1.csv'), encoding='utf-8-sig', index=False)