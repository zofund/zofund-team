# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 21:00:47 2020

@author: Administrator
"""
#导入相关第三方库
import pandas as pd
import numpy as np
import re
import matplotlib
import matplotlib.pyplot as plt
import jieba
from SentDict import *
from sklearn.feature_extraction.text import TfidfVectorizer
# from scipy import stats
# from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import classification_report, confusion_matrix

pd.set_option('display.max_rows',None)
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family']='sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False

#获取预处理后的数据
def get_clean_data(path):
    report = pd.read_csv(path, encoding="gbk")
    report_summary = report['report_summary']
    print ("Total Report: ", report.shape[0])
    #print(report.head())
#去除文本噪音,保留所有中文字符
    stripreport = report_summary.copy()
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    for i in range(len(stripreport)):
        stripreport[i] = re.sub(pattern,"",stripreport[i])  #  去除所有非汉字内容
    #print(stripreport.head())
    return stripreport,report_summary,report
path = "report.csv"
stripreport,report_original,report = get_clean_data(path)

all_word_list = []
all_doc_list = []
jieba.load_userdict('CSFD.txt')
#暂时使用前1000份研报进行训练
for report_summary in stripreport.head(25000):
    word_list = jieba.lcut(str(report_summary), cut_all=False)
    doc_list = " ".join(word_list)
    all_word_list.append(word_list)
    all_doc_list.append(doc_list)
    
def seg_sentence(word_list):
    #读取停用词表
    stopwords  = [word.strip() for word in open('baidu_stopwords.txt',encoding = 'UTF-8')]
    #补充停用词，研报中存在较多中性的词语，如果不处理可能会影响整体情感得分，后续仍需继续补充
    stopwords.extend(["环比","价格","建议","亿元",
            "营业收入","归母净利润","季度","净利润","百分点","为元元","为元元元","一季度","一日","一是","一个","一次性","万万","万人次","万美元","万元","万只","万吨","万头","万股","上市公司","公斤","基于","假设","时间","理由","投资建议","同比","港元"])
    #剔除停用词及长度为 1的词语
    seg_txt = [ w for w in word_list if w not in stopwords and len(w)>1]
    return seg_txt

all_word_list_after_stopwords = []
all_doc_list_after_stopwords = []
for word_list in all_word_list:
    word_list_after_stopwords = seg_sentence(word_list) 
    doc_list_after_stopwords = " ".join(word_list_after_stopwords)
    all_word_list_after_stopwords.append(word_list_after_stopwords)
    all_doc_list_after_stopwords.append(doc_list_after_stopwords)


#print(all_word_list_after_stopwords[0:10])
#print(all_doc_list_after_stopwords[0:10])
docs = all_word_list_after_stopwords

pos_dict = pd.read_csv("postive_dict.csv", encoding="gbk")
neg_dict = pd.read_csv("negative_dict.csv", encoding="gbk")
pos_list=list(pos_dict["postive"].values)
neg_list=list(neg_dict['negative'].values)

min_times = 0.001*len(docs)
sent_dict = SentDict(docs,method="PMI",min_times=min_times,scale="+-1",pos_seeds = pos_list,neg_seeds = neg_list)

tfidf_vec = TfidfVectorizer(ngram_range=(1,1),max_df = 0.95,min_df =0.001)#,max_features=num_features)
#tfidf_vec = TfidfVectorizer()
tfidf_matrix  = tfidf_vec.fit_transform(all_doc_list_after_stopwords)
tfidf_features = tfidf_vec.get_feature_names()
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),columns= tfidf_features)
print(tfidf_df)
sent_score,words =sent_dict.analyse_sent(tfidf_features,avg=False)
sent_df = pd.DataFrame(sent_score,index = words).sort_values(by = 0,ascending = False)
sent_df = sent_df.rename(columns = {0:"sent_score"})
sent_tf_idf = sent_df.join(tfidf_df[words].T)
print(sent_tf_idf)

#针对TF-IDF以及情感评分可人为对某些重点跟踪词汇赋值,也可针对重点词汇做分类
label = pd.DataFrame(index = tfidf_df.index)
label["景气"] = (tfidf_df["景气"] != 0).map(lambda x:1 if x ==True else 0) 
label["超预期"]  = (tfidf_df["超预期"] != 0).map(lambda x:1 if x ==True else 0) 
label["转型"] = (tfidf_df["转型"] != 0 ).map(lambda x:1 if x ==True else 0) 
label["市占率提升"] = ((tfidf_df["份额"] != 0)|(tfidf_df["扩张"] != 0)| (tfidf_df["规模"] != 0)|(tfidf_df["替代"] != 0)|
                  (tfidf_df["放量"] != 0)|(tfidf_df["拓展"] != 0)|(tfidf_df["集中度"] != 0)).map(lambda x:1 if x ==True else 0) 
label["产品价格上涨"] = ((tfidf_df["提价"] != 0) ).map(lambda x:1 if x ==True else 0) 

print(label)

num_of_report =3390 #研报编号索引 
print(docs[num_of_report])
sent_tf_idf_non_zero = sent_tf_idf[sent_tf_idf[num_of_report]!=0] #剔除TF-IDF为0的词语
x = list(sent_tf_idf_non_zero["sent_score"])
y = list(sent_tf_idf_non_zero[num_of_report])
score = sum(sent_tf_idf_non_zero["sent_score"]*sent_tf_idf_non_zero[num_of_report])

score_norm = 10/(1+np.e**(-score)) #通过sigmod函数将评分标准化到0-10，0-5表示偏负面情绪，5-10表示偏积极情绪

print(score_norm)
print(sent_tf_idf_non_zero.loc[:,["sent_score",num_of_report]].sort_values(by = "sent_score",ascending = False))
print(report_original.loc[num_of_report])
word_list = list(sent_tf_idf_non_zero.index)
plt.scatter(x,y,alpha=0.5)
plt.xlabel("单词情感得分")
plt.ylabel("TF-IDF权重")
for i in range(len(x)):
    plt.annotate(word_list[i], xy = (x[i], y[i]))

score_list =[]
for num_of_report in range(5000):
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

#rng = np.random.RandomState(0)
#indices = np.arange(len(tfidf_df))
#rng.shuffle(indices)
from sklearn.model_selection import train_test_split
#from imblearn.over_sampling import SMOTE # 过抽样处理库SMOTE
test_num = 5000   # 选择800篇文本信息
X = tfidf_df.loc[:test_num] # 随机从完整数据集中挑选800篇
y = label.iloc[:test_num,2]
#model_smote = SMOTE() # 建立SMOTE模型对象
#X_smote_resampled, y_smote_resampled = model_smote.fit_sample(X,y) # 输入数据并作过抽样处理
X_smote_resampled, y_smote_resampled = X,y
print("success rate:",y_smote_resampled.sum()/y_smote_resampled.count()) # 打印输出经过SMOTE处理后的数据集样本分类分布
print("num of sample:",len(X_smote_resampled))

texts = report_original.loc[indices[:test_num ]]
X_train, X_test, y_train, y_test = train_test_split(X_smote_resampled, y_smote_resampled, test_size = 0.2, random_state = 628)
#images = np.array(newsgroup_train.data)[indices[:test_num]]
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
#from imblearn.under_sampling import RandomUnderSampler # 欠抽样处理库RandomUnderSampler
from sklearn.svm import SVC #SVM中的分类算法SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Fitting a logistic regression model with default parameters
logreg = GradientBoostingClassifier()
logreg.fit(X_train, y_train)
# Prediction & Evaluation
y_hat_test = logreg.predict(X_test)
# Logistic Regression score
print("Logistic regression score for test set:", round(logreg.score(X_test, y_test), 5))
print("\confusion_matrix report:")
print(confusion_matrix(y_test, y_hat_test))
print("\classification_report report:")
print(classification_report(y_test, y_hat_test))