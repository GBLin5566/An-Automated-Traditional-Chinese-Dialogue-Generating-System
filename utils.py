"""Data utils"""

class Lang:
    def __init__(self):
        self.word2index = {"SOS": 0, "EOS": 1, "COS": 2}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "COS"}
        self.n_words = 3

    def build_dict(self, sentence):
        for word in sentence:
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word[self.n_words] = word
                self.n_words += 1
            else:
                self.word2count[word] += 1
