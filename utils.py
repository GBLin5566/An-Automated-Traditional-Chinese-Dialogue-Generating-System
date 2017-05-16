"""Data utils"""

import torch.cuda
from torch.autograd import Variable

import json

from tqdm import tqdm

class Lang:
    def __init__(self):
        self.word2index = {"SOS": 0, "EOS": 1, "COS": 2, "EOD": 3}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "COS", 3: "EOD"}
        self.n_words = 4

    def build_dict(self, document):
        for sentences in tqdm(document, desc='Building dict'):
            for sentence in sentences:
                for word in sentence:
                    word = word.lower()
                    if word not in self.word2index:
                        self.word2index[word] = self.n_words
                        self.word2count[word] = 1
                        self.index2word[self.n_words] = word
                        self.n_words += 1
                    else:
                        self.word2count[word] += 1

    def prune_dict(self, threshold=2):
        print("Before prune dict size ", len(self.word2index))
        prune_num = 0
        to_be_pruned = []
        for key in self.word2index.keys():
            if key in self.word2count and self.word2count[key] < threshold:
                to_be_pruned.append(key)
                prune_num += 1
        to_be_pruned = set(to_be_pruned)

        new_word2index = {}
        new_index2word = {}
        self.n_words = 0
        for key in self.word2index.keys():
            if key in to_be_pruned:
                continue
            new_word2index[key] = self.n_words
            new_index2word[self.n_words] = key
            self.n_words += 1

        self.word2index = new_word2index
        self.index2word = new_index2word

        print("Prune ", prune_num, " words")
        print("After prune dict size ", len(self.word2index))

    def sentence2index(self, sentence):
        indexs = []
        indexs.append(self.word2index["SOS"])
        for word in sentence:
            if word in self.word2index:
                indexs.append(self.word2index[word])
        indexs.append(self.word2index["EOS"])
        return indexs

def build_lang(json_path, dump_torch_variable=True):
    with open(json_path, 'r') as jsonfile:
        whole_list = json.load(jsonfile)
    my_lang = Lang()
    my_lang.build_dict(whole_list)
    my_lang.prune_dict()

    document_list = []
    for sentences in tqdm(whole_list, desc='Indexing & Making torch variable'):
        dialog = []
        for sentence in sentences:
            if dump_torch_variable:
                sentence = Variable(torch.LongTensor(my_lang.sentence2index(\
                        sentence)))
                if torch.cuda.is_available():
                    sentence = sentence.cuda()
            dialog.append(sentence)
        # Make dialog end with a special mark EOD
        end_of_dialog = []
        eod_var = Variable(torch.LongTensor(my_lang.sentence2index(\
                ["EOD"])))
        if torch.cuda.is_available():
            eod_var = eod_var.cuda()
        dialog.append(eod_var)
        document_list.append(dialog)
    return my_lang, document_list
