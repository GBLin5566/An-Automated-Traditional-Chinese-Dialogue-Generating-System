# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import range

import utils

import argparse
import time
import os
import sys 
import random
import math
import json
import codecs

import numpy as np

import utils
from utils import check_cuda_for_var, check_directory 

parser = argparse.ArgumentParser(description=\
        "Dialog2Vec Generator")
parser.add_argument('--data', type=str,\
        help='location of the data corpus(json file)')
parser.add_argument('--validation_p', type=float, default=0.2,
        help='percentage of validation data / all data')
parser.add_argument('--seed', type=int, default=55665566,
        help='random seed')
parser.add_argument('--only_stat', type=bool, default=False,
        help='only do statistic or not')
args = parser.parse_args()

random.seed(args.seed)

my_lang, document_list = utils.build_lang(args.data, dump_torch_variable=False)
# Statistic
dialog_len_count = {}
sentence_count = 0
total_word_count = 0
word_count = {}
for dialog in document_list:
    dialog_len = len(dialog)
    sentence_count += dialog_len
    for sentence in dialog:
        total_word_count += len(sentence)
        for index in sentence:
            word = my_lang.index2word[index]
            word_count[word] = word_count.setdefault(word, 0) + 1
    dialog_len_count[dialog_len] = dialog_len_count.setdefault(dialog_len, 0) + 1
print("total_word_count ", total_word_count)
print("sentence_count ", sentence_count)
print("dialog_len_count ", dialog_len_count)
print("word_count ", word_count)
if args.only_stat:
    sys.exit(0)
#
random.shuffle(document_list)
cut = int(len(document_list) * args.validation_p)
training_data, validation_data = \
        document_list[cut:], document_list[:cut]
# Training data for doc2vec
print("Training data for doc2vec")
gensim_train = []
for train_dialog in training_data:
    doc = []
    for sentence in train_dialog[:-1]:
        doc += sentence
    gensim_train.append(doc)
np.save("label/gensim_train.npy", gensim_train)

print("Label data for training")
label = []
dialog2vec = []
doc2vec = []
for train_dialog in training_data:
    doc = []
    dialog = []
    for sentence in train_dialog:
        if not sentence == train_dialog[-1]:
            doc += sentence
        if len(sentence) > 1:
            dialog.append(my_lang.index2sentence(sentence[:-1]))
    dialog2vec.append(dialog[:-1])
    doc2vec.append(doc)
    label.append(1)
    doc = []
    dialog = []
    for sentence in train_dialog[:random.randint(1, len(train_dialog)-2)]:
        doc += sentence
        if len(sentence) > 1:
            dialog.append(my_lang.index2sentence(sentence[:-1]))
    dialog2vec.append(dialog[:-1])
    doc2vec.append(doc)
    label.append(0)

np.save("label/gensim_train_test.npy", doc2vec)
np.save("label/train_label.npy", label)
with codecs.open("label/dialog2vec_train.json", "w+", encoding="utf-8") as outfile:
    json.dump(dialog2vec, outfile, indent=4, ensure_ascii=False)

print("Label data for testing")
label = []
dialog2vec = []
doc2vec = []
for validate_dialog in validation_data:
    doc = []
    dialog = []
    for sentence in validate_dialog:
        if not sentence == train_dialog[-1]:
            doc += sentence
        if len(sentence) > 1:
            dialog.append(my_lang.index2sentence(sentence[:-1]))
    dialog2vec.append(dialog[:-1])
    doc2vec.append(doc)
    label.append(1)
    doc = []
    dialog = []
    for sentence in validate_dialog[:random.randint(1, len(validate_dialog)-2)]:
        doc += sentence
        if len(sentence) > 1:
            dialog.append(my_lang.index2sentence(sentence[:-1]))
    dialog2vec.append(dialog[:-1])
    doc2vec.append(doc)
    label.append(0)

np.save("label/gensim_test_test.npy", doc2vec)
np.save("label/test_label.npy", label)
with codecs.open("label/dialog2vec_test.json", "w+", encoding="utf-8") as outfile:
    json.dump(dialog2vec, outfile, indent=4, ensure_ascii=False)
