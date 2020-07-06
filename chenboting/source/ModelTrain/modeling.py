#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/18 下午10:25
# @Author : Boting Chen
# @File : models.py
import torch

import torch.nn as nn
from torch.nn import MSELoss, BCEWithLogitsLoss
from transformers import (
    BertModel,
    BertPreTrainedModel,
    RobertaForSequenceClassification,
    XLNetForSequenceClassification,
    AlbertForSequenceClassification
)
from source.LocalEnv import LABEL_COLS


class BertForMultiClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = len(LABEL_COLS)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,  # 输入的id,模型会帮你把id转成embedding
            attention_mask=None,  # attention里的mask
            token_type_ids=None,  # [CLS]A[SEP]B[SEP] 就这个A还是B, 有的话就全1, 没有就0
            position_ids=None,  # 位置id
            head_mask=None,  # 哪个head需要被mask掉
            inputs_embeds=None,  # 可以选择不输入id,直接输入embedding
            labels=None,  # 做分类时需要的label
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output, pooled_output = outputs[:2]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits, pooled_output, sequence_output)

        if labels is not None:

            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs  # loss, logits, pooled_output, sequence_output


class RobertaForMultiLabelSequenceClassification(RobertaForSequenceClassification):
    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
    ):
        outputs = self.roberta(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()

            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1, self.num_labels)
            )
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class XLNetForMultiLabelSequenceClassification(XLNetForSequenceClassification):
    def forward(
        self,
        input_ids,
        token_type_ids=None,
        input_mask=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        labels=None,
        head_mask=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            head_mask=head_mask,
        )
        output = transformer_outputs[0]

        output = self.sequence_summary(output)
        logits = self.logits_proj(output)

        # Keep mems, hidden states, attentions if there are in it
        outputs = (logits,) + transformer_outputs[1:]

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()

            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1, self.num_labels)
            )
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class AlbertForMultiLabelSequenceClassification(AlbertForSequenceClassification):
    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
    ):
        outputs = self.albert(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()

            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1, self.num_labels)
            )
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

