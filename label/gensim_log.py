from sklearn.utils import shuffle
from sklearn import linear_model
import numpy as np

train_x = np.load("gensim_train_vector.npy")
train_y = np.load("train_label.npy")

test_x = np.load("gensim_test_vector.npy")
test_y = np.load("test_label.npy")

model = linear_model.LogisticRegression()
model.fit(train_x, train_y)

print("Gensim score: ", model.score(test_x, test_y))
