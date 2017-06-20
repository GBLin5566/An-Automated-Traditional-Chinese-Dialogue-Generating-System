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
parser.add_argument('--beam', type=int, default=1,
        help='beam size for beam search(default 1 will be greedy search)')
parser.add_argument('--eodlong', type=int, default=0,
        help='whether force model to gen a longer dialog (1 for on, 0 for off, default = 0)')
parser.add_argument('--nosr', type=int, default=0,
        help='whether force model don\'t self repeat (1 for on, 0 for off, default = 0)')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)

DEBUG = False

if args.type != "hrnn" and args.type != "seq2seq":
    raise ValueError("args.type should be hrnn or seq2seq, but got %s" % (args.type))
if args.beam <= 0:
    raise ValueError("args.beam should be at least 1 or larger number")
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
        max_dialog_len = 20
        max_sentence_len = 15
        beam_size = args.beam
        for _ in range(max_dialog_len):
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
            # Beam search
            index2state = {}
            for index in range(beam_size):
                index2state[index] = [decoder_input, decoder_hidden, [decoder_input.data[0][0]], 0.0]
            # One step to get beam_size candidates
            decoder_output, decoder_hidden = decoder(context_hidden,\
                    decoder_input, decoder_hidden)
            scores, topi = decoder_output.data.topk(beam_size)
            for index in range(beam_size):
                ni = topi[0][index]
                index2state[index][0] = check_cuda_for_var(Variable(torch.LongTensor([[ni]])))
                index2state[index][1] = decoder_hidden
                index2state[index][2].append(ni)
                index2state[index][3] = scores[0][index]
            for sentence_pointer in range(max_sentence_len):
                current_scores = []
                current2state = {}
                # Init current2state
                for index in range(beam_size):
                    for jndex in range(beam_size):
                        current2state[index * beam_size + jndex] = [0, 0, 0, 0]
                for index in range(beam_size):
                    output, hidden = decoder(context_hidden, \
                            index2state[index][0], index2state[index][1])
                    tops, topi = output.data.topk(beam_size)
                    for jndex in range(beam_size):
                        ni = topi[0][jndex]
                        current_map = current2state[index * beam_size + jndex]
                        current_map[0] = check_cuda_for_var(Variable(torch.LongTensor([[ni]])))
                        current_map[1] = hidden
                        current_map[2] = index2state[index][2][:]
                        current_map[2].append(ni)
                        current_map[3] = tops[0][jndex] + index2state[index][3]
                        if args.eodlong == 1 and my_lang.word2index["EOD"] in current_map[2]:
                            current_map[3] *= ((2*max_sentence_len - sentence_pointer) / max_sentence_len)
                        current_scores.append(current_map[3])

                _, top_of_beamsize2 = torch.FloatTensor(current_scores).topk(beam_size)
                # Top beam's output is eos, break and output the top beam
                if current2state[top_of_beamsize2[0]][2][-1] == my_lang.word2index["EOS"]:
                    if args.nosr == 1 and current2state[top_of_beamsize2[0]][2] in talking_history:
                        # Don't repeat itself
                        current2state[top_of_beamsize2[0]][3] *= 2
                    else:
                        first_eos = current2state[top_of_beamsize2[0]][2].index(my_lang.word2index["EOS"])
                        gen_sentence = current2state[top_of_beamsize2[0]][2][:first_eos+1]
                        break
                after_beam_dict = {}
                for index, candidate in enumerate(top_of_beamsize2):
                    after_beam_dict[index] = current2state[candidate]
                index2state = after_beam_dict
            # Beam Search a good sentence and assign to gen_sentence
            talking_history.append(gen_sentence)
            gen_sentence = Variable(torch.LongTensor(gen_sentence))
            gen_sentence = check_cuda_for_var(gen_sentence)
            # TODO input "訂餐廳" will get a error
            string = ' '.join([my_lang.index2word[word.data[0]] for word in gen_sentence])
            print(string)
            if "EOD" in string:
                break
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
        max_length = 20
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
                if DEBUG:
                    print("[Debug] ", decoder_input.data)
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
