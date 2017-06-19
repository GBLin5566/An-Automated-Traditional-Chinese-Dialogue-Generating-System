# -*- coding: utf-8 -*-
"""An-Automated-Traditional-Chinese-Dialogue-Generating-System Main file"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import range

import argparse
import time
import os
import sys
import random
import math
import json

import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable

import model
from ops_seq2seq import train, validate, sample
import utils
from utils import check_cuda_for_var, check_directory 

parser = argparse.ArgumentParser(description=\
        'Pytorch Traditional Chinese Dialouge Generating System builded on Hierarchical RNN.')

parser.add_argument('--data', type=str,
        help='location of the data corpus(json file)')
parser.add_argument('--validation_p', type=float, default=0.2,
        help='percentage of validation data / all data')
parser.add_argument('--embedsize', type=int, default=250,
        help='size of word embeddings')
parser.add_argument('--encoder_hidden', type=int, default=250,
        help='number of hidden units per layer in encoder')
parser.add_argument('--decoder_hidden', type=int, default=250,
        help='number of hidden units per layer in decoder')
parser.add_argument('--encoder_layer', type=int, default=2,
        help='number of layers in encoder')
parser.add_argument('--decoder_layer', type=int, default=2,
        help='number of layers in decoder')
parser.add_argument('--tie', dest='tie', action='store_true',
        help='tie the weight of embedding and output linear')
parser.add_argument('--no-tie', dest='tie', action='store_false',
        help='don\'t tie the weight of embedding and output linear')
parser.set_defaults(tie=True)
parser.add_argument('--lr', type=float, default=0.001,
        help='initial learning rate')
parser.add_argument('--clip', type=float, default=5.0,
        help='gradient clipping')
parser.add_argument('--epochs', type=int, default=1000,
        help='upper epoch limit')
parser.add_argument('--dropout', type=float, default=0.25,
        help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=55665566,
        help='random seed')
parser.add_argument('--teacher', dest='teacher', action='store_true',
        help='teacher force')
parser.add_argument('--no-teacher', dest='teacher', action='store_false',
        help='no teacher force')
parser.set_defaults(teacher=True)
parser.add_argument('--ss', dest='ss', action='store_true',
        help='scheduled sampling')
parser.add_argument('--no-ss', dest='ss', action='store_false',
        help='no scheduled sampling')
parser.set_defaults(ss=True)
parser.add_argument('--save', type=str, default='model/',
        help='path to save the final model\'s directory')
parser.add_argument('--test', dest='test', action='store_true',
        help='test mode')
parser.set_defaults(test=False)
parser.add_argument('--limit', type=int, default=0,
        help='limit the size of whole data set')
parser.add_argument('--startepoch', type=int, default=0,
        help='epoch\'s number when starting(for scheduled sampling\'s ratio)')
parser.add_argument('--restore', dest='restore', action='store_true',
        help='Reload the saved model')
parser.set_defaults(restore=False)
args = parser.parse_args()


torch.manual_seed(args.seed)
random.seed(args.seed)

check_directory(args.save)
# Read data
my_lang, document_list = utils.build_lang(args.data)
max_length = 20
random.shuffle(document_list)
if args.limit != 0:
    document_list = document_list[:args.limit]
cut = int(len(document_list) * args.validation_p)
training_data, validation_data = \
        document_list[cut:], document_list[:cut]
# Test mode
if args.test:
    # Load last model
    number = torch.load(os.path.join(args.save, 'checkpoint.pt'))
    encoder = torch.load(os.path.join(args.save, 'encoder'+str(number)+'.pt'))
    decoder = torch.load(os.path.join(args.save, 'decoder'+str(number)+'.pt'))
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    
    for dialog in validation_data:
        sample(my_lang, dialog, encoder, decoder, max_length)
        time.sleep(3)

    sys.exit(0)

learning_rate = args.lr
criterion = nn.NLLLoss()
if not args.restore:
    encoder = model.EncoderRNN(len(my_lang.word2index), args.encoder_hidden, \
            args.encoder_layer, args.dropout)
    decoder = model.DecoderRNNSeq(args.decoder_hidden, len(my_lang.word2index), \
            args.decoder_layer, args.dropout, max_length)
else:
    print("Load last model in %s" % (args.save))
    number = torch.load(os.path.join(args.save, 'checkpoint.pt'))
    encoder = torch.load(os.path.join(args.save, 'encoder'+str(number)+'.pt'))
    decoder = torch.load(os.path.join(args.save, 'decoder'+str(number)+'.pt'))
if torch.cuda.is_available():
    print("Make encoder & decoder cuda")
    encoder = encoder.cuda()
    encoder.is_cuda = True
    decoder = decoder.cuda()
    decoder.is_cuda = True
    criterion = criterion.cuda()
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

if args.tie:
    # Tying two Embedding matrix and output Linear layer
    # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
    # https://arxiv.org/abs/1611.01462
    encoder.embedding.weight = decoder.embedding.weight = decoder.out.weight

since = time.time()

best_validation_score = 10000
patient = 10
model_number = 0
teacher_lazy_period = 40
if args.teacher:
    teacher_forcing_ratio = 1.
else:
    teacher_forcing_ratio = 0.

# Save info. for loss.
save_training_loss = []
save_validation_loss = []

def save_loss(train, val):
    with open(os.path.join(args.save, "loss.json"), "w") as outfile:
        json.dump([train, val], outfile)

for epoch in range(args.startepoch + 1, args.epochs + 1):
    training_loss = 0
    iter_since = time.time()
    try:
        for index, dialog in enumerate(training_data):
            if args.ss:
                teacher_forcing_ratio = (teacher_lazy_period - epoch + 1) / teacher_lazy_period
                if teacher_forcing_ratio < 0.5:
                    teacher_forcing_ratio = 0.5
            training_loss += train(my_lang, criterion, teacher_forcing_ratio,\
                    dialog, encoder, decoder, \
                    encoder_optimizer, decoder_optimizer, max_length)
            if (index) % 100 == 0:
                print("    @ Iter [", index + 1, "/", len(training_data),"] | avg. loss: ", training_loss / (index + 1), \
                        " | perplexity: ", math.exp(training_loss / (index + 1))," | usage ", time.time() - iter_since, " seconds | teacher_force: ", \
                        teacher_forcing_ratio)
                sample(my_lang, dialog, encoder, decoder, max_length)
                iter_since = time.time()
            if (index + 1) % 2000 == 0:
                val_since = time.time()
                validation_score_100 = validate(my_lang, criterion,
                        validation_data[:100], encoder, decoder, max_length)
                print("    @ Val. [", index + 1, "/", len(training_data),"] | avg. val. loss: ", validation_score_100, \
                        " | perplexity: ", math.exp(validation_score_100)," | usage ", time.time() - val_since, " seconds")
                print("    % Best validation score: ", best_validation_score)
                if validation_score_100 < best_validation_score:
                    best_validation_score = validation_score_100
                    patient = 5
                elif patient > 0:
                    patient -= 1
                else:
                    print("****Learining rate decay****")
                    learning_rate /= 2.
                    patient = 10
                    best_validation_score = validation_score_100
                print("    % After validation best validation score: ", best_validation_score)
        validation_score = validate(my_lang, criterion, \
                validation_data, encoder, decoder, max_length)
        save_training_loss.append(training_loss / (index + 1))
        save_validation_loss.append(validation_score)
        save_loss(save_training_loss, save_validation_loss)
        print("# ", epoch, " | ", time.time() - since," seconds | validation loss: ", validation_score, " | validation perplexity: ", \
                math.exp(validation_score))
        since = time.time()
        model_number += 1
        print("Saving better model number ",model_number)
        best_validation_score = validation_score
        torch.save(encoder, os.path.join(args.save, "encoder" + str(model_number) + ".pt"))
        torch.save(decoder, os.path.join(args.save, "decoder" + str(model_number) + ".pt"))
        torch.save(model_number, os.path.join(args.save, "checkpoint.pt"))
    except ValueError:
        print(sys.exc_info())
        model_number += 1
        print("Get stopped, saving the latest model")
        torch.save(encoder, os.path.join(args.save, "encoder" + str(model_number) + ".pt"))
        torch.save(decoder, os.path.join(args.save, "decoder" + str(model_number) + ".pt"))
        torch.save(model_number, os.path.join(args.save, "checkpoint.pt"))
        break

