# -*- coding: utf-8 -*-
"""An-Automated-Traditional-Chinese-Dialogue-Generating-System Main file"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import os

import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable

import model
import utils

parser = argparse.ArgumentParser(description=\
        'Pytorch Traditional Chinese Dialouge Generating System builded on Hierarchical RNN.')

parser.add_argument('--data', type=str,
        help='location of the data corpus(json file)')
parser.add_argument('--embedsize', type=int, default=200,
        help='size of word embeddings')
parser.add_argument('--encoder_hidden', type=int, default=200,
        help='number of hidden units per layer in encoder')
parser.add_argument('--context_hidden', type=int, default=200,
        help='number of hidden units per layer in context rnn')
parser.add_argument('--decoder_hidden', type=int, default=200,
        help='number of hidden units per layer in decoder')
parser.add_argument('--encoder_layer', type=int, default=2,
        help='number of layers in encoder')
parser.add_argument('--context_layer', type=int, default=2,
        help='number of layers in context')
parser.add_argument('--decoder_layer', type=int, default=2,
        help='number of layers in decoder')
parser.add_argument('--lr', type=float, default=0.001,
        help='initial learning rate')
parser.add_argument('--clip', type=float, default=5.0,
        help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
        help='upper epoch limit')
parser.add_argument('--dropout', type=float, default=0.2,
        help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=5566,
        help='random seed')
parser.add_argument('--save', type=str, default='model/',
        help='path to save the final model\'s directory')

args = parser.parse_args()

torch.manual_seed(args.seed)

def check_cuda_for_var(var):
    if torch.cuda.is_available():
        return var.cuda()
    else:
        return var

def check_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Read data
my_lang, document_list = utils.build_lang(args.data)

learning_rate = args.lr
encoder = model.EncoderRNN(my_lang.n_words, args.encoder_hidden, \
        args.encoder_layer, args.dropout)
context = model.ContextRNN(args.encoder_hidden * args.encoder_layer, args.context_hidden, \
        args.context_layer, args.dropout)
decoder = model.DecoderRNN(args.context_hidden * args.context_layer, args.decoder_hidden, \
        my_lang.n_words, args.decoder_layer, args.dropout)
if torch.cuda.is_available():
    encoder = encoder.cuda()
    context = context.cuda()
    decoder = decoder.cuda()
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
context_optimizer = optim.Adam(context.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

def train(training_data):
    # Zero gradients
    encoder_optimizer.zero_grad()
    context_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = Variable(torch.FloatTensor(1))
    nn.init.constant(loss, 0)
    loss = check_cuda_for_var(loss)

    context_hidden = context.init_hidden()

    predict_count = 0

    for index, sentence in enumerate(training_data):
        if index == len(training_data) - 1:
            break
        decoder_input = Variable(torch.LongTensor([[0]]))
        decoder_input = check_cuda_for_var(decoder_input)
        encoder_hidden = encoder.init_hidden()
        decoder_hidden = decoder.init_hidden()
        for ei in range(len(sentence)):
            _, encoder_hidden = encoder(sentence[ei], encoder_hidden)
        encoder_hidden = encoder_hidden.view(1, 1, -1)
        context_output, context_hidden = context(encoder_hidden, context_hidden)
        next_sentence = training_data[index+1]
        for di in range(len(next_sentence)):
            decoder_output, decoder_hidden = decoder(context_hidden,\
                    decoder_input, decoder_hidden)
            loss += criterion(decoder_output[0], next_sentence[di])
            decoder_input = next_sentence[di].unsqueeze(1)
            predict_count += 1

    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), args.clip)
    torch.nn.utils.clip_grad_norm(context.parameters(), args.clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), args.clip)
    encoder_optimizer.step()
    context_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / (predict_count)

since = time.time()
for epoch in range(1, args.epochs + 1):
    training_loss = 0
    iter_since = time.time()
    for index, dialog in enumerate(document_list):
        training_loss += train(dialog)
        if (index + 1) % 10 == 0:
            print("    @ Iter [", index + 1, "/", len(document_list),"] | loss: ", training_loss / (index + 1), \
                    " | usage ", time.time() - iter_since, " seconds")
            iter_since = time.time()
    if epoch % 10  == 0:
        print("# ", epoch, " | ", time.time() - since," seconds | loss: ", training_loss)
        since = time.time()
