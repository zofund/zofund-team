# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:00:27 2020

@author: Administrator
"""

#导入相关第三方库
import pandas as pd
import numpy as np
import re
import matplotlib
#import matplotlib.pyplot as plt
import jieba
from SentDict import *
from sklearn.feature_extraction.text import TfidfVectorizer

pd.set_option('display.max_rows',None)
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family']='sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False

#获取预处理后的数据
def get_clean_data(path):
    report = pd.read_csv(path, encoding="gbk")
    report_summary = report['report_summary']
    print ("Total Report: ", report.shape[0])

#去除文本噪音,保留所有中文字符
    stripreport = report_summary.copy()
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    for i in range(len(stripreport)):
        stripreport[i] = re.sub(pattern,"",stripreport[i])  #  去除所有非汉字内容

    return stripreport,report_summary,report
path = "report.csv"
stripreport,report_original,report = get_clean_data(path)


def cut_words(texts):
    all_word_list = []
    jieba.load_userdict('CSFD.txt')
    for text in texts:
        word_list = jieba.lcut(str(text), cut_all=False)
        all_word_list.append(word_list)
    all_doc_list = [" ".join(word_listd) for word_listd in all_word_list]  
    return all_word_list,all_doc_list
    
all_word_list,all_doc_list = cut_words(stripreport)

def remove_stopwords(all_word_list):
    #读取停用词表
    all_word_list_after_stopwords = []
    stopwords  = [word.strip() for word in open('baidu_stopwords.txt',encoding = 'UTF-8')]
    #去除长度为1的单词
    for word_list in all_word_list:
        word_list_after_stopwords = [ w for w in word_list if w not in stopwords and len(w)>1]    
    all_word_list_after_stopwords.append(word_list_after_stopwords)
    all_doc_list_after_stopwords= [" ".join(word_list_after_stopwords) for word_list_after_stopwords in all_word_list_after_stopwords]

    return all_word_list_after_stopwords,all_doc_list_after_stopwords


all_word_list_after_stopwords,all_doc_list_after_stopwords = remove_stopwords(all_word_list)
docs = all_word_list_after_stopwords

pos_dict = pd.read_csv("postive_dict.csv", encoding="gbk")
neg_dict = pd.read_csv("negative_dict.csv", encoding="gbk")
pos_list=list(pos_dict["postive"].values)
neg_list=list(neg_dict['negative'].values)

min_times = 0.005*len(docs)
sent_dict = SentDict(docs,method="PMI",min_times=min_times,scale="+-1",pos_seeds = pos_list,neg_seeds = neg_list)

tfidf_vec = TfidfVectorizer(ngram_range=(1,1),max_df = 0.95,min_df =0.05)
tfidf_matrix  = tfidf_vec.fit_transform(all_doc_list_after_stopwords)
tfidf_features = tfidf_vec.get_feature_names()
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),columns= tfidf_features)
print(tfidf_df)
sent_score,words =sent_dict.analyse_sent(tfidf_features,avg=False)
sent_df = pd.DataFrame(sent_score,index = words).sort_values(by = 0,ascending = False)
sent_df = sent_df.rename(columns = {0:"sent_score"})
sent_tf_idf = sent_df.join(tfidf_df[words].T)
print(sent_tf_idf)


score_list = []
for num_of_report in range(len(stripreport)):
    sent_tf_idf_non_zero = sent_tf_idf[sent_tf_idf[num_of_report]!=0] #剔除TF-IDF为0的词语
    x = list(sent_tf_idf_non_zero["sent_score"])
    y = list(sent_tf_idf_non_zero[num_of_report])
    score = sum(sent_tf_idf_non_zero["sent_score"]*sent_tf_idf_non_zero[num_of_report])
    score_norm = 10/(1+np.e**(-score)) #通过sigmod函数将评分标准化到0-10
    score_list.append(score_norm)

print(score_list)

#情感评分分布，明显偏向积极情绪
pd.DataFrame(score_list).hist(bins=10)
#情感得分排序
pd.DataFrame(score_list).sort_values(by = 0)