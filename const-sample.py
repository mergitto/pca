import numpy as np

def to_float64(vector_list):
    return np.asarray(vector_list, dtype=np.float64)

model = Word2Vec.load("Your word2vec model name")

# 入力に用いた単語（文章）の分散表現を↓に追加
SHIKAKU = to_float64(model['input word here'])


