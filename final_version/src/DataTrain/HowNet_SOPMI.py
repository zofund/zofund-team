#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/7 10:51
# @Author  : Boting CHEN
# @Site    : 
# @File    : HowNet_SOPMI.py
# @Software: PyCharm

import os
import re
import jieba
import gensim
import OpenHowNet
import numpy as np
import pandas as pd
import tushare as ts
import jieba.posseg as psg
from tqdm import tqdm
from collections.abc import Iterable
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from src.LocalEnv import INPUT_PATH, OUTPUT_PATH, SAVED_MODEL_PATH
from src.Utils.Logging import LOGGER


class SentDict(object):
    """
    #统计两个词语在文本中同时出现的概率，如果概率越大，其相关性就越紧密，关联度越高。
    """

    def __init__(self, docs=[], method="PMI", min_times=5, scale="None", pos_seeds=None, neg_seeds=None):
        super(SentDict, self).__init__()
        self.sent_dict = {}
        self.words = set()
        assert isinstance(pos_seeds, Iterable)
        assert isinstance(neg_seeds, Iterable)
        self.build_sent_dict(docs, method, min_times, scale, pos_seeds, neg_seeds)

    def __getitem__(self, key):
        return self.sent_dict[key]

    @property
    def pos_seeds(self, _pos_seeds):
        _pos_seeds = set(_pos_seeds) & set(self.words)
        return _pos_seeds

    @property
    def neg_seed(self, neg_seeds):
        _neg_seeds = set(neg_seeds) & set(self.words)
        return _neg_seeds

    def build_sent_dict(self, docs=[], method="PMI", min_times=5, scale="None", pos_seeds=None, neg_seeds=None):
        self.doc_count = len(docs)
        self.method = method
        pos_seeds = set(pos_seeds)
        neg_seeds = set(neg_seeds)
        if self.doc_count > 0:
            if method == "PMI":
                self.co_occur, self.one_occur = self.get_word_stat(docs)
                self.words = set(word for word in self.one_occur if self.one_occur[word] >= min_times)
                if len(pos_seeds) > 0 or len(neg_seeds) > 0:  # 如果有新的输入，就更新种子词，否则默认已有（比如通过set已设定）
                    self.pos_seeds = (pos_seeds & self.words)
                    self.neg_seeds = (neg_seeds & self.words)
                if len(self.pos_seeds) > 0 or len(self.neg_seeds) > 0:
                    self.sent_dict = self.SO_PMI(self.words, scale)
                else:
                    raise Exception("你的文章中不包含种子词，SO-PMI算法无法执行")
            else:
                raise Exception("不支持的情感分析算法")

    def analyse_sent(self, words, avg=True):
        if self.method == "PMI":
            words = (set(words) & set(self.sent_dict))
            if avg:
                return sum(self.sent_dict[word] for word in words) / len(words) if len(words) > 0 else 0
            else:
                return [self.sent_dict[word] for word in words], words
        else:
            raise Exception("不支持的情感分析算法")

    @staticmethod
    def get_word_stat(docs, co=True):
        co_occur = dict()  # 由于defaultdict太占内存，还是使用dict
        one_occur = dict()
        for doc in docs:
            for word in doc:
                if not word in one_occur:
                    one_occur[word] = 1
                else:
                    one_occur[word] += 1
                # 考虑自共现，否则如果一个负面词不与其他负面词共存，那么它就无法获得PMI，从而被认为是负面的，这不合情理
                if not (word, word) in co_occur:
                    co_occur[(word, word)] = 1
                else:
                    co_occur[(word, word)] += 1
            if co:
                for a, b in combinations(doc, 2):
                    if not (a, b) in co_occur:
                        co_occur[(a, b)] = 1
                        co_occur[(b, a)] = 1
                    else:
                        co_occur[(a, b)] += 1
                        co_occur[(b, a)] += 1
        return co_occur, one_occur

    def PMI(self, w1, w2):
        if not ((w1 in self.one_occur) and (w2 in self.one_occur)):
            raise Exception()
        if not (w1, w2) in self.co_occur:
            return 0
        c1, c2 = self.one_occur[w1], self.one_occur[w2]
        c3 = self.co_occur[(w1, w2)]
        return np.log2((c3 * self.doc_count) / (c1 * c2))

    def SO_PMI(self, words, scale="None"):
        ret = {}
        max0, min0 = 0, 0
        for word in words:
            tmp = sum(self.PMI(word, seed) for seed in self.pos_seeds) - \
                  sum(self.PMI(word, seed) for seed in self.neg_seeds)
            max0, min0 = max(tmp, max0), min(tmp, min0)
            ret[word] = tmp
        if scale == "+-1":
            # 在正负两个区域分别做线性变换
            # 不采用统一线性变换2*(x-mid)/(max-min)的原因:
            # 要保留0作为中性情感的语义，否则当原来的最小值为0时，经过变换会变成-1
            for word, senti in ret.items():
                if senti > 0:  # 如果触发此条件，max0≥senti>0, 不用检查除数为0。下同
                    ret[word] /= max0
                elif senti < 0:
                    ret[word] /= (-min0)
        elif scale == "0-10":
            # 这里可以采用同一变换
            ret = {word: 10 * (senti - min0) / (max0 - min0) for word, senti in ret.items()}
        return ret


class SOPMIModel:
    def __init__(self):
        self.report_path = os.path.join(INPUT_PATH, 'clean_data.csv')
        self.csfd_path = os.path.join(INPUT_PATH, 'CSFD.txt')
        self.stopwords_path = os.path.join(INPUT_PATH, 'baidu_stopwords.txt')
        self.pos_list = pd.read_csv(os.path.join(INPUT_PATH, 'postive_dict.csv'), encoding='gbk'
                                    ).values.flatten().tolist()
        self.neg_list = pd.read_csv(os.path.join(INPUT_PATH, 'negative_dict.csv'), encoding='gbk'
                                    ).values.flatten().tolist()

    def get_clean_data(self):
        report = pd.read_csv(self.report_path, encoding="utf-8-sig")
        print("Total Report: {}".format(len(report)))
        # 只保留所有中文字符
        pattern = re.compile(r'[^\u4e00-\u9fa5]')
        stripreport = report['report_summary'].apply(lambda x: re.sub(pattern, "", x))

        return stripreport, report

    def cut_words(self, texts):
        all_word_list = []
        jieba.load_userdict(self.csfd_path)
        for text in texts:
            word_list = jieba.lcut(str(text), cut_all=False)
            all_word_list.append(word_list)
        all_doc_list = [" ".join(word_listd) for word_listd in all_word_list]
        return all_word_list, all_doc_list

    def remove_stopwords(self, all_word_list):
        # 读取停用词表
        all_word_list_after_stopwords = []
        stopwords = [word.strip() for word in open(self.stopwords_path, encoding='UTF-8')]
        stopwords.extend(self.stk_names)
        # 去除长度为1的单词
        for word_list in all_word_list:
            word_list_after_stopwords = [w for w in word_list if w not in stopwords and len(w) > 1]
            all_word_list_after_stopwords.append(word_list_after_stopwords)
        all_doc_list_after_stopwords = [" ".join(word_list_after_stopwords) for word_list_after_stopwords in
                                        all_word_list_after_stopwords]

        return all_word_list_after_stopwords, all_doc_list_after_stopwords

    @property
    def stk_names(self):
        _stk_names = ts.get_stock_basics().name.tolist()
        return _stk_names

    def run(self, re_train_sopmi=False):
        """
        :param re_train_sopmi: 是否重新训练sopmi模型, 建议不要，如果重新训练建议使用至少64G内存的机器，否则可能内存不够。
        :return:
        """
        if re_train_sopmi:
            stripped_report, report = self.get_clean_data()
            all_word_list, all_doc_list = self.cut_words(stripped_report)
            all_word_list_after_stopwords, all_doc_list_after_stopwords = self.remove_stopwords(all_word_list)
            docs = all_word_list_after_stopwords

            min_times = 0.005 * len(docs)
            sent_dict = SentDict(docs, method="PMI", min_times=min_times, scale="+-1", pos_seeds=self.pos_list,
                                 neg_seeds=self.neg_list)

            tf_idf_vec = TfidfVectorizer(ngram_range=(1, 1), max_df=0.95, min_df=0.05)
            tf_idf_matrix = tf_idf_vec.fit_transform(all_doc_list_after_stopwords)
            tf_idf_features = tf_idf_vec.get_feature_names()
            tf_idf_df = pd.DataFrame(tf_idf_matrix.toarray(), columns=tf_idf_features)
            sent_score, words = sent_dict.analyse_sent(tf_idf_features, avg=False)
            sent_df = pd.DataFrame(sent_score, index=words).sort_values(by=0, ascending=False)
            sent_df = sent_df.rename(columns={0: "sent_score"})
            sent_tf_idf = sent_df.join(tf_idf_df[words].T)
        else:
            sent_tf_idf = pd.read_csv(os.path.join(INPUT_PATH, 'sent_tf_idf.csv'), encoding='utf-8-sig', index_col=0)
            cols = list(map(lambda x: int(x), list(sent_tf_idf.columns[1:])))
            sent_tf_idf.columns = ['sent_score'] + cols

        sent_score_mat = sent_tf_idf.iloc[:, 0].to_frame().values.T
        sent_tf_idf_mat = sent_tf_idf.iloc[:, 1:].values
        scores = np.dot(sent_score_mat, sent_tf_idf_mat).flatten().tolist()
        scores = list(map(lambda x: 10 / (1 + np.e ** (-x)), scores))

        res = pd.Series(scores)
        res.columns = ['sent_score']
        res.to_csv(os.path.join(OUTPUT_PATH, 'sopmi_result.csv'))


class HowNetModel:
    def __init__(self):
        self.report_path = os.path.join(INPUT_PATH, 'clean_data.csv')
        self.saved_word2vec_path = os.path.join(SAVED_MODEL_PATH, 'model.bin')
        self.jieba_user_dict_path = os.path.join(INPUT_PATH, 'userdict.txt')
        self.report = self._load_report()
        self.model = self._load_model()
        self.hownet_dict = self._load_hownet_dict()
        self._init_jieba()

    def _load_report(self):
        """
        载入数据，进行句子切分
        :return:
        """
        report = pd.read_csv(self.report_path, encoding='utf-8-sig')
        report['sentence_summary'] = report['report_summary'].map(lambda i: re.split('[：。；，]', i))
        LOGGER.info('report loaded')
        return report

    def _load_model(self):
        """
        载入word2vec模型
        :return:
        """
        model = gensim.models.Word2Vec.load(self.saved_word2vec_path)
        LOGGER.info('word2vec model loaded')
        return model

    def _init_jieba(self):
        jieba.load_userdict(self.jieba_user_dict_path)

    @staticmethod
    def _load_hownet_dict():
        LOGGER.info('引入hownet')
        hownet_dict = OpenHowNet.HowNetDict()
        LOGGER.info('引入hownet计算相似度')
        hownet_dict.initialize_sememe_similarity_calculation()
        LOGGER.info('引入完毕')
        return hownet_dict

    def word_score_max(self, word_list, compare_word_list):
        """
        计算一个词与一个词组里面的相似度(基于word2vec)，并返回相似度的最大的词和相似度得分
        通过'word'索引到单词，通过'score'索引到得分
        :param word_list:
        :param word:
        :return:
        """
        compare = []
        for word in compare_word_list:
            word_score_dict = {}
            for i in word_list:
                try:
                    c = self.model[i]
                    score = self.model.similarity(word, i)
                    word_score_dict[i] = score
                except KeyError:
                    c = 0
            if word_score_dict == {}:
                return {'word': 0, 'score': 0}
            else:
                max_word = max(word_score_dict, key=word_score_dict.get)
                dict_ = {'word': max_word, 'score': word_score_dict[max_word]}
            compare.append(dict_['score'])
        return np.max(compare)

    def word_score_v(self, word_list, word_compare_list):
        """
        计算单句动词或其他词性与褒贬程度词的相似性（基于hownet）
        返回相似度
        :param word_list:
        :param word_compare_list:
        :return:
        """
        compare = []
        for k in word_compare_list:
            word_score_dict = {}
            for i in word_list:
                try:
                    score = self.hownet_dict.calculate_word_similarity(i, k)
                    word_score_dict[i] = score
                except KeyError:
                    c = 0
            if word_score_dict == {}:
                dict_ = {'word': 0, 'score': 0}
            else:
                max_word = max(word_score_dict, key=word_score_dict.get)
                dict_ = {'word': max_word, 'score': word_score_dict[max_word]}
            compare.append(dict_['score'])
        return np.max(compare)

    def word_location(self, sentences, word):
        """
        遍历给定文档中的句子，确定与归因最接近的句子，定位该句子
        :param sentences: 
        :param word: 
        :return: 
        """
        score_grade = {}
        for sentence in sentences:
            n_list = []
            v_list = []
            adj_list = []
            adv_list = []

            for i in psg.cut(sentence):
                if i.flag == 'n':
                    n_list.append(i.word)
                elif i.flag == 'v':
                    v_list.append(i.word)
                elif i.flag == 'a':
                    adj_list.append(i.word)
                elif i.flag == 'd':
                    adv_list.append(i.word)

            score_grade[sentence] = self.word_score_max(n_list, word)['score']
        grade_sentence = max(score_grade, key=score_grade.get)
        return grade_sentence

    @staticmethod
    def specific_sentence(sentence):
        """
        对给定单句进行词性分割
        返回一个字典，可以通过索引来获得给定词性的词列表
        :param sentence:
        :return:
        """
        n_sentence_list = []
        v_sentence_list = []
        adj_sentence_list = []
        adv_sentence_list = []
        all_list = []

        for i in psg.cut(sentence):
            all_list.append(i.word)
            if i.flag == 'n' or i.flag == 'vn':
                n_sentence_list.append(i.word)
            elif i.flag == 'v':
                v_sentence_list.append(i.word)
            elif i.flag == 'a':
                adj_sentence_list.append(i.word)
            elif i.flag == 'd':
                adv_sentence_list.append(i.word)

        data_ = {
            'n_list': n_sentence_list,
            'v_list': v_sentence_list,
            'adj_list': adj_sentence_list,
            'adv_list': adv_sentence_list,
            'all_list': all_list
        }

        return data_

    @staticmethod
    def adv_deal_word(word):
        """
         输入一个词，如果为修正副词，取出其修正度，如果不是，为
        :param word:
        :return:
        """
        data_adv = pd.read_csv(r'adv.txt', encoding='utf-8', sep=' ', header=None)

        if word in data_adv[0].values:
            print(word)
            return eval(data_adv[data_adv[0].values == word].iloc[0, 1])
        else:
            return 1

    @staticmethod
    def adv_deal_sentence(sentence):
        # 输入一个句子，得出该句子副词修正度
        spec_sent = HowNetModel.specific_sentence(sentence)
        if len(spec_sent['adv_list']) != 0:
            # print('有副词')
            for word in spec_sent['adv_list']:
                # print(HowNetModel.adv_deal_word(word))
                return HowNetModel.adv_deal_word(word)
        else:
            # print('无副词')
            return 1

    def score(self, sentence, special_word, degree):
        """
        对给定句子进行情感打分，并给定阈值参数
        :param sentence:
        :param special_word:
        :param degree:
        :return:
        """
        spec_sent = self.specific_sentence(sentence)
        if self.word_score_max(spec_sent['n_list'], special_word)['score'] > degree:
            print(sentence)
            # 取出该句子的所有不同词性的几何
            v_list = spec_sent['v_list']
            n_list = spec_sent['n_list']
            adv_list = spec_sent['adv_list']
            adj_list = spec_sent['adj_list']
            all_list = spec_sent['all_list']

            # 把评分分为三档，好，中，坏，好为1分
            score_good = self.word_score_v(
                v_list + n_list + adj_list,
                ['提高', '高', '增加', '好', '成功', '上涨', '超', '增量']
            )

            score_mediem = self.word_score_v(
                all_list,
                ['稳健', '持续', '维持', '稳定', '保持', '符合']
            )

            score_bad = self.word_score_v(
                v_list + n_list + adj_list,
                ['降低', '低', '低于', '减弱', '坏', '失败']
            )

            dict_ = {
                'good': score_good,
                'mediem': score_mediem,
                'bad': score_bad
            }

            mood = max(dict_, key=dict_.get)
            if dict_[mood] > 0.7:
                if mood == 'good':
                    score = 1
                elif mood == 'mediem':
                    score = 0.5
                elif mood == 'bad':
                    score = -1
            # score = dict[max(dict,key=dict.get)]
                print(mood)
                print(score * HowNetModel.adv_deal_sentence(sentence))
                return score * HowNetModel.adv_deal_sentence(sentence)
            else:
                print('无明显情感倾向', 0)
                return 0
        else:
            print(sentence)
            print(0)
            return 0

    def run(self):
        achievement = []
        degree_of_boom = []
        market_share = []
        price = []
        transformation = []
        i = 1
        # 遍历文档
        for sentenses in tqdm(self.report['sentence_summary']):
            score_grade = {}
            # 定位句子

            achievement_sentence = self.word_location(sentenses, ['业绩'])
            degree_of_boom_sentence = self.word_location(sentenses, ['行业景气度','量价','产业','行业'])
            market_share_sentence = self.word_location(sentenses, ['市场占有率','体量','规模','集中度','市占率','市场份额','占率'])
            price_sentence = self.word_location(sentenses, ['产品价格','价格','批价','出厂价','量价'])
            transformation_sentence = self.word_location(sentenses, ['转型','转变'])

            print('--------')
            print(i)
            # 情感打分
            achievement.append(self.score(achievement_sentence, '业绩', 0.7))
            degree_of_boom.append(self.score(price_sentence, '价格', 0.65))
            market_share.append(self.score(degree_of_boom_sentence, '行业景气度', 0.6))
            price.append(self.score(market_share_sentence, '市场占有率', 0.6))
            transformation.append(self.score(transformation_sentence, '转型', 0.6))
            print('\n')
            # 把最接近归因的句子按词性分割
            i = i + 1

        self.report['业绩'] = achievement
        self.report['产品价格'] = degree_of_boom
        self.report['行业景气度'] = market_share
        self.report['市场占有率'] = price
        self.report['转型'] = transformation
        self.report['总分'] = self.report.loc[:, '业绩': '转型'].sum(axis=1) + 5
        # self.report['总分'].hist(range=[0, 10])
        self.report.to_csv('report_score.csv', encoding='utf-8-sig')


if __name__ == '__main__':
    # _goal = 'm01'
    _goal = 'm02'
    if _goal == 'm01':
        m01 = SOPMIModel()
        m01.run(False)
    else:
        m02 = HowNetModel()
        m02.run()

    try:
        df1 = pd.read_csv(os.path.join(OUTPUT_PATH, 'sopmi_result.csv'))
        df2 = pd.read_csv('report_score.csv', encoding='utf-8-sig')
        df1['最终得分'] = df2.iloc[:,1]
        df1.to_csv('report_score.csv', encoding='utf-8-sig')
    except:
        print('缺一张表，无法合并')

