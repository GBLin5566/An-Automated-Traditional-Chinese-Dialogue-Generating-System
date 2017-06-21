
from collections import namedtuple

import numpy as np
import gensim

document = np.load("gensim_train.npy")
docs = []

for text in document:
    words = [str(w) for w in text]
    docs.append(words)

model = gensim.models.Word2Vec(docs, size=500, min_count=1, workers=4)
#model = gensim.models.Word2Vec.load("word2vec.model")

for epoch in range(20):
    if epoch % 1000 == 0 and not epoch == 0:
        model.save("word2vec.model")
    print("epoch %s" % epoch)
    model.train(docs, total_examples=model.corpus_count, epochs=model.iter)

model.save("word2vec.model")

train_document = np.load("gensim_train_test.npy")
test_document = np.load("gensim_test_test.npy")
train_vecs = []
test_vecs = []
for j, text in enumerate(train_document):
    train_vec = []
    for word in text:
        str_word = str(word)
        if str_word in model.wv:
            train_vec.append(model.wv[str_word])

    train_vecs.append(np.sum(train_vec, axis=0)/len(train_vec))
for k, text in enumerate(test_document):
    test_vec = []
    for word in text:
        str_word = str(word)
        if str_word in model.wv:
            test_vec.append(model.wv[str_word])

    test_vecs.append(np.sum(test_vec, axis=0)/len(test_vec))

print("train_vector shape ", np.shape(train_vecs))
print("test_vector shape ", np.shape(test_vecs))

np.save("word2vec_train_vector.npy", train_vecs)
np.save("word2vec_test_vector.npy", test_vecs)
