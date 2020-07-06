from gensim import models
import gensim
from gensim.models import word2vec
import pandas as pd
model = gensim.models.Word2Vec.load("model.bin")
df = pd.read_csv('word_split.txt')
# for i in df:
#     for k in i.split(' '):
#         if model.similarity("业绩",k) >= 0.7:
#             print('业绩，{}:相似度为{}'.format(k,model.similarity("业绩",k)))

df = df.dropna()
for i in df.iloc[:,0]:



    list = [k for k in str(i).split(' ')]

    score = model.n_similarity(list,['业绩','超','预期'])
    if score >= 0.7:
        print('{}-----\n{}'.format(score,list))

# import numpy as np
# from scipy import spatial
#
# index2word_set = set(model.index2word)
#
# def avg_feature_vector(sentence, model, num_features, index2word_set):
#     words = sentence.split()
#     feature_vec = np.zeros((num_features, ), dtype='float32')
#     n_words = 0
#     for word in words:
#         if word in index2word_set:
#             n_words += 1
#             feature_vec = np.add(feature_vec, model[word])
#     if (n_words > 0):
#         feature_vec = np.divide(feature_vec, n_words)
#     return feature_vec