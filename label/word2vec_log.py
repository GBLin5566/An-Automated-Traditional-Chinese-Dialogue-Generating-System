from sklearn.utils import shuffle
from sklearn import linear_model
import numpy as np

train_x = np.load("word2vec_train_vector.npy")
train_y = np.load("train_label.npy")

test_x = np.load("word2vec_test_vector.npy")
test_y = np.load("test_label.npy")

model = linear_model.LogisticRegression()
model.fit(train_x, train_y)

print("Word2vec train score: ", model.score(train_x, train_y))
print("Word2vec test score: ", model.score(test_x, test_y))
