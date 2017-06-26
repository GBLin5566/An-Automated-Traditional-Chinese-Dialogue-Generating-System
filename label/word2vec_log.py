from sklearn.utils import shuffle
from sklearn import linear_model
import numpy as np

train_x = np.load("word2vec_train_vector.npy")
train_y = np.load("train_label.npy")

test_x = np.load("word2vec_test_vector.npy")
test_y = np.load("test_label.npy")

model = linear_model.LogisticRegression()
model.fit(train_x, train_y)

print("len of train_x ", np.sum(np.linalg.norm(train_x, axis=1))/len(train_x))
print("len of test_x ", np.sum(np.linalg.norm(test_x, axis=1))/len(test_x))

print("Word2vec train score: ", model.score(train_x, train_y))
print("Word2vec test score: ", model.score(test_x, test_y))
