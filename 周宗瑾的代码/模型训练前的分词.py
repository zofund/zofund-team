import pandas as pd
import re
import jieba

jieba.load_userdict('userdict.txt')
data = pd.read_csv(r'report.csv',encoding='gb18030')
data['split_word'] = data['report_summary'].map(lambda i: re.sub(r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+", "", i))
data['split_word'] = data['split_word'].apply(lambda i: ' '.join(jieba.cut(i)))
print(data['split_word'])
data['split_word'].to_csv('word_split.txt',encoding='utf-8',index=None)

