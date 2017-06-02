# -*- coding: utf-8 -*-
"""Generator for model"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import range

import argparse
import os
import sys
import random
import pickle

import torch
from torch.autograd import Variable
# Import my own cleaning lib, use jieba for other user
try:
    from purewords import clean_sentence as clean
except ImportError:
    from jieba import lcut as clean
import model
import utils
from utils import check_cuda_for_var, check_directory

parser = argparse.ArgumentParser(description=\
        "Generator for HRNN/Seq2seq")
parser.add_argument('--data', type=str,
        help="location of the data corpus(json file)")
parser.add_argument('--type', type=str,
        help="generate dialog with hrnn/seq2seq model")
parser.add_argument('--save', type=str, default='model/',
        help='path to load the final model\'s directory')
parser.add_argument('--seed', type=int, default=55665566,
        help='random seed')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)

if args.type != "hrnn" and args.type != "seq2seq":
    raise ValueError("args.type should be hrnn or seq2seq, but got %s" % (args.type))
if not os.path.isfile('dict.pkl'):
    my_lang, _ = utils.build_lang(args.data)
    with open('dict.pkl', 'wb') as filename:
        pickle.dump(my_lang, filename)
else:
    print("Load dict.pkl")
    with open('dict.pkl', 'rb') as filename:
        my_lang = pickle.load(filename)
if args.type == "hrnn":
    # Load last HRNN model
    number = torch.load(os.path.join(args.save, 'checkpoint.pt'))
    encoder = torch.load(os.path.join(args.save, 'encoder'+str(number)+'.pt'))
    context = torch.load(os.path.join(args.save, 'context'+str(number)+'.pt'))
    decoder = torch.load(os.path.join(args.save, 'decoder'+str(number)+'.pt'))
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        context = context.cuda()
        decoder = decoder.cuda()
    def gen(sentence):
        encoder.eval()
        context.eval()
        decoder.eval()

        # Inference
        gen_sentence = []
        talking_history = []
        context_hidden = context.init_hidden()
        counter = 0
        while counter < 10:
            decoder_input = Variable(torch.LongTensor([[my_lang.word2index["SOS"]]]))
            decoder_input = check_cuda_for_var(decoder_input)
            encoder_hidden = encoder.init_hidden()
            decoder_hidden = decoder.init_hidden()
            if len(gen_sentence) > 0:
                for ei in range(len(gen_sentence)):
                    _, encoder_hidden = encoder(gen_sentence[ei], encoder_hidden)
                # Clean generated sentence list
                gen_sentence = []
            else:
                for ei in range(len(sentence)):
                    _, encoder_hidden = encoder(sentence[ei], encoder_hidden)
            context_output, context_hidden = context(encoder_hidden, context_hidden)
            # TODO Beam search
            while True:
                gen_sentence.append(decoder_input.data[0][0])
                if gen_sentence[-1] == my_lang.word2index["EOS"] or len(gen_sentence) > 15:
                    break
                decoder_output, decoder_hidden = decoder(context_hidden,\
                        decoder_input, decoder_hidden)
                _, topi = decoder_output.data.topk(1)
                ni = topi[0][0]
                decoder_input = Variable(torch.LongTensor([[ni]]))
                decoder_input = check_cuda_for_var(decoder_input)
            gen_sentence = Variable(torch.LongTensor(gen_sentence))
            gen_sentence = check_cuda_for_var(gen_sentence)
            string = ' '.join([my_lang.index2word[word.data[0]] for word in gen_sentence])
            print(string)
            talking_history.append(string)
            if "EOD" in string:
                break
            counter += 1
        return talking_history
else:
    # Load last Seq2seq model
    number = torch.load(os.path.join(args.save, 'checkpoint.pt'))
    encoder = torch.load(os.path.join(args.save, 'encoder'+str(number)+'.pt'))
    decoder = torch.load(os.path.join(args.save, 'decoder'+str(number)+'.pt'))
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    def gen(sentence):
        max_length = 16
        encoder.eval()
        decoder.eval()
        talking_history = []
        gen_sentence = []
        counter = 0
        while counter < 10:
            encoder_hidden = encoder.init_hidden()
            encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
            decoder_input = Variable(torch.LongTensor([[my_lang.word2index["SOS"]]]))
            encoder_outputs = check_cuda_for_var(encoder_outputs)
            decoder_input = check_cuda_for_var(decoder_input)
            if len(gen_sentence) > 0:
                for ei in range(len(gen_sentence)):
                    encoder_output, encoder_hidden = encoder(gen_sentence[ei], encoder_hidden)
                    encoder_outputs[ei] = encoder_output[0][0]
                    # Clean generated sentence list
                gen_sentence = []
            else:
                for ei in range(len(sentence)):
                    encoder_output, encoder_hidden = encoder(sentence[ei], encoder_hidden)
                    encoder_outputs[ei] = encoder_output[0][0]
            decoder_hidden = encoder_hidden
            while True:
                gen_sentence.append(decoder_input.data[0][0])
                if gen_sentence[-1] == my_lang.word2index["EOS"] or len(gen_sentence) >= max_length - 1:
                    break
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, \
                        encoder_outputs)
                _, topi = decoder_output.data.topk(1)
                ni = topi[0][0]
                decoder_input = Variable(torch.LongTensor([[ni]]))
                decoder_input = check_cuda_for_var(decoder_input)
            gen_sentence = Variable(torch.LongTensor(gen_sentence))
            gen_sentence = check_cuda_for_var(gen_sentence)
            string = ' '.join([my_lang.index2word[word.data[0]] for word in gen_sentence])
            print(string)
            talking_history.append(string)
            if "EOD" in string:
                break
            counter += 1
        return talking_history
# Generating string
try:
    while True:
        start = input("[%s] >>> " % (args.type.upper()))
        clean_sentence = clean(start)
        clean_sentence_idx = my_lang.sentence2index(clean_sentence)
        clean_sentence_idx = Variable(torch.LongTensor(clean_sentence_idx))
        clean_sentence_idx = check_cuda_for_var(clean_sentence_idx)
        gen(clean_sentence_idx)
except KeyboardInterrupt:
    print()
