import logging
from gensim import models
import gensim
from gensim.models import word2vec
import numpy as np
from scipy import spatial
from gensim.models.word2vec import LineSentence, Word2Vec,Text8Corpus
sentences = Text8Corpus("word_split.txt")
print(sentences)
model = Word2Vec(sentences, min_count=1)
model.save("model.bin")