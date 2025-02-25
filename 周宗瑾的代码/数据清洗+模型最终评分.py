import pandas as pd
import re
import jieba.posseg as psg
import jieba
import gensim
import numpy as np


data = pd.read_csv(r'report.csv', encoding='gb18030').iloc[:10000,:]
data['report_summary'] = data['report_summary'].map(lambda i: re.sub('<AI(.*?)>', '', i))
data['report_summary'] = data['report_summary'].map(lambda i: re.sub('</AI>', '', i))

# 按句号分割句子再分词
data['sentence_summary'] = data['report_summary'].map(lambda i: re.split('：|。|；|，', i))

# 加载word2vec模型
print('开始下载模型')
model = gensim.models.Word2Vec.load(r'model.bin')
print('下载完毕模型')

# 加载特定词词典
jieba.load_userdict('userdict.txt')

import OpenHowNet
print('引入hownet')
hownet_dict = OpenHowNet.HowNetDict()
print('引入hownet计算相似度')
hownet_dict.initialize_sememe_similarity_calculation()
print('引入完毕')
def word_score_max(word_list,word):
    # 计算一个词与一个词组里面的相似度(基于word2vec)，并返回相似度的最大的词和相似度得分
    # 通过'word'索引到单词，通过'score'索引到得分

    word_score_dict = {}
    for i in word_list:
        try:
            c = model[i]
            score = model.similarity(word, i)
            word_score_dict[i] = score
        except KeyError:
            c = 0
    if word_score_dict == {}:
        return {'word':0,'score':0}
    else:
        max_word = max(word_score_dict,key=word_score_dict.get)
        dict = {'word':max_word,'score':word_score_dict[max_word]}
        return dict



def word_score_v(word_list,word_compare_list):
    # 计算单句动词获其他词性与褒贬程度词的相似性（基于hownet）
    # 返回相似度

    compare = []
    for k in word_compare_list:
        word_score_dict = {}
        for i in word_list:
            try:
                score = hownet_dict.calculate_word_similarity(i, k)
                word_score_dict[i] = score
            except KeyError:
                c = 0
        if word_score_dict == {}:
            dict = {'word':0,'score':0}
        else:
            max_word = max(word_score_dict,key=word_score_dict.get)
            dict = {'word':max_word,'score':word_score_dict[max_word]}
        compare.append(dict['score'])
    return np.max(compare)



def word_location(sentenses,word):
    # 遍历给定文档中的句子，确定与归因最接近的句子，定位该句子

    score_grade = {}
    for sentence in sentenses:
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

        score_grade[sentence] = word_score_max(n_list, word)['score']
    grade_sentence = max(score_grade,key=score_grade.get)
    return grade_sentence




def specific_sentence(sentence):
    # 对给定单句进行词性分割
    # 返回一个字典，可以通过索引来获得给定词性的词列表

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
    data = {'n_list':n_sentence_list,'v_list':v_sentence_list,'adj_list':adj_sentence_list,'adv_list':adv_sentence_list,'all_list':all_list}
    return data


def adv_deal_word(word):
    # 输入一个词，如果为修正副词，取出其修正度，如果不是，为1
    data_adv = pd.read_csv(r'adv.txt', encoding='utf-8', sep=' ', header=None)

    if word in data_adv[0].values:
        print(word)
        return eval(data_adv[data_adv[0].values == word].iloc[0, 1])
    else:
        return 1


def adv_deal_sentence(sentence):
    # 输入一个句子，得出该句子副词修正度
    if len(specific_sentence(sentence)['adv_list']) != 0:
        print('有副词')
        for word in specific_sentence(sentence)['adv_list']:
            print(adv_deal_word(word))
            return adv_deal_word(word)
    else:
        print('无副词')
        return 1





def score(sentence,special_word,degree):
    # 对给定句子进行情感打分，并给定阈值参数

    if word_score_max(specific_sentence(sentence)['n_list'], special_word)['score'] > degree:
        print(sentence)

        #取出该句子的所有不同词性的几何
        v_list = specific_sentence(sentence)['v_list']
        n_list = specific_sentence(sentence)['n_list']
        adv_list = specific_sentence(sentence)['adv_list']
        adj_list = specific_sentence(sentence)['adj_list']
        all_list = specific_sentence(sentence)['all_list']

        # 把评分分为三档，好，中，坏，好为1分
        score_good = word_score_v(v_list+ n_list + adj_list,['提高','高','增加','好','成功','上涨','超','增量','高于','超'])
        score_mediem = word_score_v(all_list, ['稳健','持续','维持','稳定','保持','符合'])
        score_bad = word_score_v(v_list + n_list + adj_list, ['降低','低','低于','减弱','坏','失败',])


        dict = {'good':score_good,'mediem':score_mediem,'bad':score_bad}
        mood = max(dict,key=dict.get)
        if dict[mood] > 0.7:
            if mood == 'good':
                score = 1
            elif mood == 'mediem':
                score = 0.5
            elif mood == 'bad':
                score = -1
        # score = dict[max(dict,key=dict.get)]
            print(mood)
            print(score * adv_deal_sentence(sentence))
            return score * adv_deal_sentence(sentence)
        else:
            print('无明显情感倾向',0)
            return 0

    else:
        print(sentence)
        print(0)
        return 0



def main():

    achievement = []
    degree_of_boom = []
    market_share = []
    price = []
    transformation = []
    i = 1
    # 遍历文档
    for sentenses in data['sentence_summary']:
        score_grade = {}
        # 定位句子

        achievement_sentence = word_location(sentenses, '业绩')
        degree_of_boom_sentence = word_location(sentenses, '行业景气度')
        market_share_sentence = word_location(sentenses,'市场占有率')
        price_sentence = word_location(sentenses,'产品价格')
        transformation_sentence = word_location(sentenses,'转型')

        print('**************************************')
        print(i)
        # 情感打分
        achievement.append(score(achievement_sentence,'业绩',0.7))
        print('---------------')
        degree_of_boom.append(score(price_sentence,'价格',0.65))
        print('---------------')
        market_share.append(score(degree_of_boom_sentence,'行业景气度',0.6))
        print('---------------')
        price.append(score(market_share_sentence,'市场占有率',0.6))
        print('---------------')
        transformation.append(score(transformation_sentence,'转型',0.6))
        print('---------------')
        print('\n')
        # 把最接近归因的句子按词性分割
        i = i + 1


    # data['业绩'] = achievement
    # data['产品价格'] = degree_of_boom
    # data['行业景气度'] = market_share
    # data['市场占有率'] = price
    # data['转型'] = transformation
    # data['总分'] = data['业绩'] + data['产品价格'] + data['行业景气度'] + data['市场占有率'] + data['转型'] + 5
    # data['总分'].hist(range=[0,10])
    # data.to_csv(r'report_score.csv',encoding='gb18030')

main()
