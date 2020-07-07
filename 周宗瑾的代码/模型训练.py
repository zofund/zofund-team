
from gensim.models.word2vec import LineSentence, Word2Vec,Text8Corpus
sentences = Text8Corpus("word_split.txt")
print(sentences)
model = Word2Vec(sentences, min_count=1)
model.save("model.bin")