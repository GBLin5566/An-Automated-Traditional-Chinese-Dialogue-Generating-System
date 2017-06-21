
from collections import namedtuple

import numpy as np
from gensim.models import doc2vec

document = np.load("gensim_train.npy")
docs = []

analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for i, text in enumerate(document):
    words = [str(w) for w in text]
    tags = [i]
    docs.append(analyzedDocument(words, tags))

#model = doc2vec.Doc2Vec(docs, size = 500,  window = 300, min_count = 1, workers = 4)
model = doc2vec.Doc2Vec.load("gensim.model")

for epoch in range(100):
    if epoch % 1000 == 0 and not epoch == 0:
        model.save("gensim.model")
    print("epoch %s" % epoch)
    model.train(docs, total_examples=model.corpus_count, epochs=model.iter)

model.save("gensim.model")

train_document = np.load("gensim_train_test.npy")
test_document = np.load("gensim_test_test.npy")
train_vecs = []
test_vecs = []
for j, text in enumerate(train_document):
    words = [str(w) for w in text]
    train_vecs.append(model.infer_vector(words))
for k, text in enumerate(test_document):
    words = [str(w) for w in text]
    test_vecs.append(model.infer_vector(words))

print("train_vector shape ", np.shape(train_vecs))
print("test_vector shape ", np.shape(test_vecs))

np.save("gensim_train_vector.npy", train_vecs)
np.save("gensim_test_vector.npy", test_vecs)
