import psycopg2
import pandas as pd
from sqlalchemy import create_engine
import re
import jieba
import jieba.posseg as psg

# postgres config
postgres_host = "www.biogeat.com"           # 数据库地址
postgres_port = "8899"                     # 数据库端口
postgres_user = "intern_read"              # 数据库用户名
postgres_password = "intern_6677"       # 数据库密码
postgres_datebase = "intern"            # 数据库名字
postgres_table1 = "intern_test"           #数据库中的表的名字

# connection string
conn_string = "host=" + postgres_host + " port=" + postgres_port + " dbname=" + postgres_datebase + " user=" + postgres_user + " password=" + postgres_password

conn = psycopg2.connect(conn_string)
print('连接成功')

sql_command1 = "select * from public.intern_test limit 50000"

data = pd.read_sql(sql_command1, conn)

data.to_csv('report.csv',encoding='gb18030')
# df1=data[data['report_summary'].str.contains('业绩超预期')]
# print(df1)

# 定义相似度
# def Jaccrad(model, reference):#terms_reference为源句子，terms_model为候选句子
#     terms_reference= jieba.cut(reference)#默认精准模式
#     terms_model= jieba.cut(model)
#     grams_reference = set(terms_reference)#去重；如果不需要就改为list
#     grams_model = set(terms_model)
#     temp=0
#     for i in grams_reference:
#         if i in grams_model:
#             temp=temp+1
#     fenmu=len(grams_model)+len(grams_reference)-temp #并集
#     jaccard_coefficient=float(temp/fenmu)#交集
#     return jaccard_coefficient





# comment = data['report_summary']
#
# data['report_summary'] = data['report_summary'].map(lambda i: re.sub('<AI(.*?)>', '',i))
# data['report_summary'] = data['report_summary'].map(lambda i: re.sub('</AI>', '',i))

# 按句号分割句子再分词
# data['sentence_summary'] = data['report_summary'].map(lambda i: re.sub(r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:？、~@#￥%……&*（）]+", "", i))
# data['sentence_summary'] = data['report_summary'].map(lambda i:i.split('。'))
# data['sentence_summary'].to_csv('sentence_split_juhao.txt',encoding='utf-8',index=None)

# for list in data['sentence_summary']:
#     for sentence in list:
#
#         sentence = ' '.join([str(i) for i in psg.cut(sentence)])
#         "{}/{},".format(x.word, x.flag)
#         print(sentence)
# n_list
# v_list
# adj_list
# adv_list
# o_list

# 直接分词
# data['split_word'] = data['report_summary'].map(lambda i: re.sub(r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+", "", i))
# data['split_word'] = data['split_word'].apply(lambda i: ' '.join(jieba.cut(i)))
# print(data['split_word'])
# data['split_word'].to_csv('word_split.txt',encoding='utf-8',index=None)
# for summary in comment:
#     summary = re.sub('<AI(.*?)>', '',summary)
#     summary = re.sub('</AI>', '',summary)
#
#     summary = re.sub(r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+", "", summary)  # 去标点符号
#
#
#     summary = jieba.cut(summary)
#     for i in summary:
#         print(i)
    # for k in i.split('，'):
    #     for h in k.split('。'):
    #         print(Jaccrad('业绩超预期', h))


