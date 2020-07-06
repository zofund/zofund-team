#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/18 下午10:58
# @Author : Boting Chen
# @File : learner.py
import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from transformers import (
    BertConfig,
    BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
from transformers.data.processors.utils import InputExample, DataProcessor, InputFeatures
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange
from source.LocalEnv import (
    BERT_BASE_CN_PATH,
    MAX_LENGTH_DEFAULT,
    LABEL_COLS,
    RANDOM_SEED,
    LABEL_LIST,
    SAVED_BERT_PATH,
    SAVED_LSTM_PATH
)
from source.ModelTrain.modeling import BertForMultiClassification
from source import LOGGER

MODEL_CLASSES = {
    "m01_bert": (BertConfig, BertTokenizer),
}

path_base_bert = {"model_name_or_path": BERT_BASE_CN_PATH,
                  "config_name": os.path.join(BERT_BASE_CN_PATH, "bert-base-chinese-config.json"),
                  "tokenizer_name": os.path.join(BERT_BASE_CN_PATH, 'bert-base-chinese-vocab.txt'),
                  }

path_saved_bert = {"model_name_or_path": SAVED_BERT_PATH,
                   "config_name": os.path.join(SAVED_BERT_PATH,),
                   "tokenizer_name": os.path.join(SAVED_BERT_PATH),
                   }


class LongTxtBertMultiLearner:
    def __init__(
            self,
            data: pd.DataFrame,  # 打好label的数据
            bert_model,
            model_path: dict,
    ):
        self.data = data
        self.bert_model = bert_model
        self.model_path = model_path
        self.max_length = MAX_LENGTH_DEFAULT
        self.batch_size_bert = 16
        self.bert_epochs = 5
        self.lr = 2e-5
        self.eps = 1e-8
        self.batch_size_lstm = 16
        self.lstm_epochs = 60
        self.num_warmup_steps = 0

        _, tokenizer_class = MODEL_CLASSES["m01_bert"]
        self.tokenizer = tokenizer_class.from_pretrained(
            self.model_path["tokenizer_name"],
            do_lower_case=True,
            cache_dir=None,
        )

    @staticmethod
    def from_pretrained_model(data, model_path):
        config_class, _ = MODEL_CLASSES["m01_bert"]
        config = config_class.from_pretrained(model_path["config_name"])
        model_class = BertForMultiClassification

        bert_model = model_class.from_pretrained(
            model_path["model_name_or_path"],
            from_tf=False,
            config=config,
            cache_dir=None,
        )

        bert_model.to('cuda')

        return LongTxtBertMultiLearner(
            data,
            bert_model,
            model_path
        )

    def _get_split(self, text, max_length=None):
        """
        切割文本，对于过长的文本，我们按照MAX_LENGTH进行切割。
        :param text:
        :param max_length
        :return:
        """
        max_length = self.max_length if max_length is None else max_length
        l_total = []
        length_adj = max_length - 50
        if len(text) // length_adj > 0:
            n = len(text) // length_adj
        else:
            n = 1
        for w in range(n):
            if w == 0:
                l_parcial = text[:max_length]
                l_total.append(l_parcial)
            else:
                if w != n - 1:
                    l_parcial = text[w * length_adj:w * length_adj + max_length]
                    l_total.append(l_parcial)
                else:
                    l_parcial = text[-max_length:]
                    l_total.append(l_parcial)
        return l_total

    @staticmethod
    def _cutting_row(df_, label_cols=LABEL_COLS):
        df_l = []  # 分割好的文本
        df_label_l = []  # 每段文本的label
        df_index_l = []  # 该段文本属于未分割前的哪个文本. 比如是3, 那就表明此段文本是属于未分割前的第4条数据
        for idx, row in df_.iterrows():
            for l in row['report_summary']:
                df_l.append(l)
                if label_cols:
                    df_label_l.append(row[LABEL_COLS[0]:])
                df_index_l.append(idx)
        return df_l, df_label_l, df_index_l

    def _data_preprocess_train(self):
        # 首先将数据进行切割
        self.data.loc[:, 'report_summary'] = self.data['report_summary'].apply(lambda x: self._get_split(x.strip()))
        # 数据一共有16个标签, ['prosperity','occupancy','transition','product_price','exceed_expectation', 'score0', ...,
        # 'score10']
        tmp = self.data.loc[:, '行业景气度回升':'业绩超预期'].astype(bool).astype(int)
        _prefix = LABEL_COLS[-1].split('_')[0]
        score_all = pd.get_dummies(self.data.iloc[:, -5:].sum(axis=1).apply(int), prefix=_prefix)
        self.data = pd.concat([self.data.loc[:, 'report_summary'], tmp], axis=1)
        self.data = pd.concat([self.data, score_all], axis=1)
        try:
            self.data.columns = ['report_summary'] + LABEL_COLS
        except Exception as e:
            LOGGER.critical('模型训练数据不均衡！')
            raise e

        train, val = train_test_split(self.data, test_size=0.2, random_state=RANDOM_SEED)
        self.data = None  # 节省内存
        train.reset_index(drop=True, inplace=True)
        val.reset_index(drop=True, inplace=True)

        train_l, train_label_l, train_index_l = self._cutting_row(train)
        val_l, val_label_l, val_index_l = self._cutting_row(val)
        self.train, self.train_index_l = train, train_index_l
        self.val, self.val_index_l = val, val_index_l
        # 生成train和val数据
        train_df = pd.concat(train_label_l, axis=1).T
        train_df['report_summary'] = train_l
        val_df = pd.concat(val_label_l, axis=1).T
        val_df['report_summary'] = val_l
        # 生成InputExamples
        self.train_InputExamples = train_df.apply(lambda x: InputExample(guid=None,
                                                                         text_a=x['report_summary'],
                                                                         text_b=None,
                                                                         label=x[LABEL_COLS].tolist()), axis=1)

        self.val_InputExamples = val_df.apply(lambda x: InputExample(guid=None,
                                                                     text_a=x['report_summary'],
                                                                     text_b=None,
                                                                     label=x[LABEL_COLS].tolist()), axis=1)

        # 生成train_dataset
        train_features = self.convert_examples_to_features(self.train_InputExamples,
                                                           self.tokenizer,
                                                           label_list=LABEL_LIST,
                                                           max_length=self.max_length)
        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        attention_mask = torch.tensor([f.attention_mask for f in train_features], dtype=torch.long)
        token_type_ids = torch.tensor([f.token_type_ids for f in train_features], dtype=torch.long)
        the_labels = torch.tensor([f.label for f in train_features], dtype=torch.float)  # multi-label
        # 这里一步每个东西都是要传到BERT模型的forward里面的, 要传哪些自己准备好
        self.train_dataset = TensorDataset(input_ids, attention_mask, token_type_ids, the_labels)

        # 生成val_dataset
        val_features = self.convert_examples_to_features(self.val_InputExamples,
                                                         self.tokenizer,
                                                         label_list=LABEL_LIST,
                                                         max_length=self.max_length)
        input_ids = torch.tensor([f.input_ids for f in val_features], dtype=torch.long)
        attention_mask = torch.tensor([f.attention_mask for f in val_features], dtype=torch.long)
        token_type_ids = torch.tensor([f.token_type_ids for f in val_features], dtype=torch.long)
        the_labels = torch.tensor([f.label for f in val_features], dtype=torch.float)  # multi-label
        self.val_dataset = TensorDataset(input_ids, attention_mask, token_type_ids, the_labels)

    def _data_preprocess_predict(self):
        """
        将测试集数据进行预处理
        :return:
        """
        # 首先将数据进行切割
        self.data.loc[:, 'report_summary'] = self.data['report_summary'].apply(lambda x: self._get_split(x.strip()))
        test_l, _, test_index_l = self._cutting_row(self.data, [])
        self.test_index_l = test_index_l
        test_df = pd.Series(test_l).to_frame()
        test_df.columns = ['report_summary']

        # 生成测试集
        self.test_InputExamples = test_df.apply(lambda x: InputExample(guid=None,
                                                                       text_a=x['report_summary'],
                                                                       text_b=None,
                                                                       label=[]), axis=1)
        # 生成test_dataset
        test_freatures = self.convert_examples_to_features(self.test_InputExamples,
                                                           self.tokenizer,
                                                           label_list=LABEL_LIST,
                                                           max_length=self.max_length)
        input_ids = torch.tensor([f.input_ids for f in test_freatures], dtype=torch.long)
        attention_mask = torch.tensor([f.attention_mask for f in test_freatures], dtype=torch.long)
        token_type_ids = torch.tensor([f.token_type_ids for f in test_freatures], dtype=torch.long)
        # the_labels = torch.tensor([f.label for f in test_freatures], dtype=torch.float)  # multi-label
        # 这里一步每个东西都是要传到BERT模型的forward里面的, 要传哪些自己准备好
        self.test_dataset = TensorDataset(input_ids, attention_mask, token_type_ids)

    def _get_bert_prediction(self, model, dataset: TensorDataset, for_predict=False):

        LOGGER.info("***** Running prediction to fetch embedding  *****")
        LOGGER.info("  Num examples = %d", len(dataset))
        LOGGER.info("  Batch size = {}".format(self.batch_size_bert))

        pooled_outputs = None

        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

        for batch in tqdm(dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to("cuda") for t in batch)

            with torch.no_grad():
                if for_predict:
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                else:
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                outputs = model(**inputs)
                pooled_output = outputs[-2]

                if pooled_outputs is None:
                    pooled_outputs = pooled_output.detach().cpu().numpy()
                else:
                    pooled_outputs = np.append(pooled_outputs, pooled_output.detach().cpu().numpy(), axis=0)

        return pooled_outputs

    def fit_bert(self):
        """
        finetune bert模型
        :return:
        """
        self._data_preprocess_train()

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.bert_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,

            },
            {
                "params": [p for n, p in self.bert_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
        ]

        t_total = len(self.train_dataset) // self.bert_epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, eps=self.eps)
        # bert里的小技巧, bert里的learning rate是不断变化的,先往上升,再往下降,这个scheduler就是用来设置这个
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.num_warmup_steps, num_training_steps=t_total
        )

        # *********************
        LOGGER.info("*****Running training*****")
        LOGGER.info("  Num examples = {}".format(len(self.train_dataset)))
        LOGGER.info("  Num Epochs = {}".format(self.bert_epochs))

        epochs_trained = 0
        global_step = 0
        steps_trained_in_current_epoch = 0

        tr_loss, logging_loss = 0.0, 0.0
        self.bert_model.zero_grad()
        train_iterator = trange(epochs_trained, self.bert_epochs, desc="Epoch", disable=False)

        for _ in train_iterator:  # 默认10个epoch

            # 随机打包
            train_sampler = RandomSampler(self.train_dataset)
            train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.batch_size_bert)
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)

            for step, batch in enumerate(epoch_iterator):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                self.bert_model.train()
                batch = tuple(t.to("cuda") for t in batch)

                # 每个batch里是 input_ids, attention_mask, token_type_ids, the_labels
                # 所以传入模型时,每个参数位置对应好放进去.
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2],
                          "labels": batch[3]}

                outputs = self.bert_model(**inputs)
                loss = outputs[0]
                # loss = loss.sigmoid()  # 由于是多标签，不要使用softmax
                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % 1 == 0:
                    torch.nn.utils.clip_grad_norm_(self.bert_model.parameters(), 1.0)

                    optimizer.step()
                    scheduler.step()
                    self.bert_model.zero_grad()
                    global_step += 1

            LOGGER.info("average loss:" + str(tr_loss / global_step))
        LOGGER.info("Global Step: {}".format(global_step))

        self.save_bert()

    def save_bert(self):
        self.bert_model.save_pretrained(SAVED_BERT_PATH)
        self.tokenizer.save_pretrained(SAVED_BERT_PATH)
        torch.save(self.model_path, os.path.join(SAVED_BERT_PATH, 'training_args.bin'))

    def validate_bert(self):
        """
        由于我们这里只需要取bert的embedding作为输入放入下一层的LSTM,故而不需要validate BERT
        :return:
        """
        raise NotImplementedError

    def fit(self, re_finetune):
        """
        首先fit_bert，然后将bert的embedding取出来，将相同一篇的文章的进行合并，再输入LSTM进行预测
        :param: re_finetune: 是否重新finetune，如果是，则重新finetune，否则载入已经finetune过的模型
        :return:
        """
        if re_finetune:
            self.fit_bert()
        else:
            self._data_preprocess_train()

        train_pooled_outputs = self._get_bert_prediction(self.bert_model, self.train_dataset)
        val_pooled_outputs = self._get_bert_prediction(self.bert_model, self.val_dataset)

        # 将embedding组合起来作为长句子的feature
        def embedding_ensemble(pooled_outputs, index_l, raw_df):
            x = {}

            for l, emb in zip(index_l, pooled_outputs):
                if l in x.keys():
                    x[l] = np.vstack([x[l], emb])
                else:
                    x[l] = [emb]

            l_final = []
            label_l_final = []
            for k in x.keys():
                l_final.append(x[k])
                label_l_final.append(raw_df.loc[k][LABEL_COLS])

            res = pd.concat(label_l_final, axis=1).T
            res['emb'] = l_final

            return res

        df_val = embedding_ensemble(val_pooled_outputs, self.val_index_l, self.val)
        df_train = embedding_ensemble(train_pooled_outputs, self.train_index_l, self.train)

        def data_generator(df):
            num_sequences = len(df['emb'])
            batch_size = self.batch_size_lstm
            batches_per_epoch = num_sequences // batch_size
            num_features = 768

            x_list = df['emb'].to_list()
            # Generate batches
            while True:
                for b in range(batches_per_epoch):
                    # longest_index = (b + 1) * batch_size - 1
                    timesteps = len(max(x_list[:(b + 1) * batch_size][-batch_size:], key=len))
                    x = np.full((batch_size, timesteps, num_features), -99.)
                    y = np.zeros((batch_size, len(LABEL_COLS)))
                    for i in range(batch_size):
                        li = b * batch_size + i
                        x[i, 0:len(x_list[li]), :] = x_list[li]
                        y[i] = df[LABEL_COLS].iloc[li, :].values
                    yield x, y

        train_generator, val_generator = data_generator(df_train), data_generator(df_val)
        self.fit_lstm(train_generator,
                      steps_per_epoch=len(df_train) // self.batch_size_lstm,
                      epochs=self.lstm_epochs,
                      validation_data=val_generator,
                      validation_steps=len(df_val) // self.batch_size_lstm,
                      callbacks=[LongTxtBertMultiLearner.call_reduce()])

    def predict(self):
        self._data_preprocess_predict()
        test_pooled_outputs = self._get_bert_prediction(self.bert_model, self.test_dataset, True)

        # 将embedding组合起来作为长句子的feature
        def embedding_ensemble(pooled_outputs, index_l):
            x = {}

            for l, emb in zip(index_l, pooled_outputs):
                if l in x.keys():
                    x[l] = np.vstack([x[l], emb])
                else:
                    x[l] = [emb]

            l_final = []
            for k in x.keys():
                l_final.append(x[k])

            res = pd.Series(l_final).to_frame()
            res.columns = ['emb']

            return res

        df_test = embedding_ensemble(test_pooled_outputs, self.test_index_l)
        lstm_model = self.create_lstm_model()
        lstm_model.load_weights(os.path.join(SAVED_LSTM_PATH, 'lstm_model.h5'))

        batch_size = 1

        def x_generator(df, batch_size=batch_size):
            num_sequences = len(df['emb'])
            batches_per_epoch = num_sequences // batch_size
            num_features = 768

            x_list = df['emb'].to_list()
            # Generate batches
            while True:
                for b in range(batches_per_epoch):
                    # longest_index = (b + 1) * batch_size - 1
                    timesteps = len(max(x_list[:(b + 1) * batch_size][-batch_size:], key=len))
                    x = np.full((batch_size, timesteps, num_features), -99.)
                    for i in range(batch_size):
                        li = b * batch_size + i
                        x[i, 0:len(x_list[li]), :] = x_list[li]
                    yield x

        predicted_ = lstm_model.predict(
            x=x_generator(df_test, batch_size=batch_size),
            steps=len(df_test) // batch_size
        )

        return predicted_

    @staticmethod
    def create_lstm_model():
        text_input = keras.Input(shape=(None, 768,), dtype='float32', name='text')

        # keras.layers.Masking(mask_value=0.0)是用于对值为指定值的位置进行掩蔽的操作，以忽略对应的timestep。
        l_mask = keras.layers.Masking(mask_value=-99.)(text_input)

        # Encoded in a single vector via a LSTM
        encoded_text = keras.layers.LSTM(100, )(l_mask)
        out_dense = keras.layers.Dense(30, activation='relu')(encoded_text)
        # 添加sigmoid layer
        out = keras.layers.Dense(len(LABEL_COLS), activation='sigmoid')(out_dense)  # 这里应该用sigmoid
        # Specify input and output
        model = keras.Model(text_input, out)
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['acc'])
        model.summary()

        return model

    @staticmethod
    def fit_lstm(train_data, steps_per_epoch, epochs, validation_data, validation_steps, callbacks):
        model = LongTxtBertMultiLearner.create_lstm_model()
        model.fit(train_data, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=validation_data,
                  validation_steps=validation_steps, callbacks=callbacks)
        model.save(os.path.join(SAVED_LSTM_PATH, 'lstm_model.h5'))

    @staticmethod
    def call_reduce(monitor='val_acc', factor=0.95, patience=3, verbose=2,
                    mode='auto', min_delta=0.01, cooldown=0, min_lr=0):
        return keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, factor=factor, patience=patience, verbose=verbose,
            mode=mode, min_delta=min_delta, cooldown=cooldown, min_lr=min_lr
        )

    @staticmethod
    def convert_examples_to_features(examples, tokenizer, label_list, max_length):
        """
        transformers的glue_convert_examples_to_features对于multi-label无法直接调用，我们这里选择自己写一个适用于多标签的converter.
        :return:
        """
        label_map = {label: i for i, label in enumerate(label_list)}

        features = []
        for (ex_index, example) in enumerate(examples):
            try:
                tokens_a = tokenizer.tokenize(example.text_a)
            except:
                print("Cannot tokenise item {}, Text:{}".format(
                    ex_index, example.text_a))

            tokens_b = None
            if example.text_b:
                tokens_b = tokenizer.tokenize(example.text_b)
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                LABEL_LIST._truncate_seq_pair(tokens_a, tokens_b, max_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_length - 2:
                    tokens_a = tokens_a[:(max_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_length
            assert len(input_mask) == max_length
            assert len(segment_ids) == max_length

            if isinstance(example.label, list):
                label_id = []
                for label in example.label:
                    label_id.append(float(label))
            else:
                if example.label is not None:
                    label_id = label_map[example.label]
                else:
                    label_id = ''

            features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=input_mask,
                              token_type_ids=segment_ids,
                              label=label_id))
        return features

    @staticmethod
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()


if __name__ == '__main__':
    from source.LocalEnv import INPUT_PATH, OUTPUT_PATH

    # _goal = 'train'
    _goal = 'test'

    if _goal == 'train':
        train_ = pd.read_csv(os.path.join(INPUT_PATH, 'train.csv'))
        learner_ = LongTxtBertMultiLearner.from_pretrained_model(train_, path_saved_bert)
        learner_.fit(False)
    else:
        test_ = pd.read_csv(os.path.join(INPUT_PATH, 'test.csv')).iloc[:100, :].reset_index()
        learner_ = LongTxtBertMultiLearner.from_pretrained_model(test_, path_saved_bert)
        res = pd.DataFrame(learner_.predict())
        res.columns = LABEL_COLS
        res = pd.concat([test_.iloc[:, :4], res], axis=1)
        res.to_csv(os.path.join(OUTPUT_PATH, 'res1.csv'), encoding='utf-8-sig', index=False)
