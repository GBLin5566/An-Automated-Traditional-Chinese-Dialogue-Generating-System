from sklearn.utils import shuffle
from sklearn import linear_model
import numpy as np

train_x = np.load("dialog2vector_train.npy")
train_x = train_x.reshape(np.shape(train_x)[0], np.shape(train_x)[1]*np.shape(train_x)[2]*np.shape(train_x)[3])
train_y = np.load("train_label.npy")

test_x = np.load("dialog2vector_test.npy")
test_x = test_x.reshape(np.shape(test_x)[0], np.shape(test_x)[1]*np.shape(test_x)[2]*np.shape(test_x)[3])
test_y = np.load("test_label.npy")

model = linear_model.LogisticRegression()
model.fit(train_x, train_y)

print("Dialog2vector score: ", model.score(test_x, test_y))
