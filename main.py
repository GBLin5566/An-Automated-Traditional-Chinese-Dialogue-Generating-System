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
parser.add_argument('--validation_p', type=float, default=0.2,
        help='percentage of validation data / all data')
parser.add_argument('--embedsize', type=int, default=100,
        help='size of word embeddings')
parser.add_argument('--encoder_hidden', type=int, default=100,
        help='number of hidden units per layer in encoder')
parser.add_argument('--context_hidden', type=int, default=100,
        help='number of hidden units per layer in context rnn')
parser.add_argument('--decoder_hidden', type=int, default=100,
        help='number of hidden units per layer in decoder')
parser.add_argument('--encoder_layer', type=int, default=2,
        help='number of layers in encoder')
parser.add_argument('--context_layer', type=int, default=2,
        help='number of layers in context')
parser.add_argument('--decoder_layer', type=int, default=2,
        help='number of layers in decoder')
parser.add_argument('--lr', type=float, default=0.005,
        help='initial learning rate')
parser.add_argument('--clip', type=float, default=5.0,
        help='gradient clipping')
parser.add_argument('--epochs', type=int, default=4,
        help='upper epoch limit')
parser.add_argument('--dropout', type=float, default=0.15,
        help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=5566,
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

args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)

def check_cuda_for_var(var):
    if torch.cuda.is_available():
        return var.cuda()
    else:
        return var

def check_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

check_directory(args.save)
# Read data
my_lang, document_list = utils.build_lang(args.data)

learning_rate = args.lr
encoder = model.EncoderRNN(my_lang.n_words, args.encoder_hidden, \
        args.encoder_layer, args.dropout)
context = model.ContextRNN(args.encoder_hidden * args.encoder_layer, args.context_hidden, \
        args.context_layer, args.dropout)
decoder = model.DecoderRNN(args.context_hidden * args.context_layer, args.decoder_hidden, \
        my_lang.n_words, args.decoder_layer, args.dropout)
# Tying two Embedding matrix and output Linear layer
# "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
# https://arxiv.org/abs/1611.01462
encoder.embedding.weight = decoder.embedding.weight = decoder.out.weight
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

    model_predict = []

    for index, sentence in enumerate(training_data):
        if index == len(training_data) - 1:
            break
        decoder_input = Variable(torch.LongTensor([[0]]))
        decoder_input = check_cuda_for_var(decoder_input)
        encoder_hidden = encoder.init_hidden()
        decoder_hidden = decoder.init_hidden()
        for ei in range(len(sentence)):
            if ei > len(model_predict) - 1 or random.random() < teacher_forcing_ratio:
                _, encoder_hidden = encoder(sentence[ei], encoder_hidden)
            else:
                _, encoder_hidden = encoder(model_predict[ei], encoder_hidden)

        encoder_hidden = encoder_hidden.view(1, 1, -1)
        context_output, context_hidden = context(encoder_hidden, context_hidden)
        next_sentence = training_data[index+1]
        model_predict = []
        for di in range(len(next_sentence)):
            predict_count += 1
            decoder_output, decoder_hidden = decoder(context_hidden,\
                    decoder_input, decoder_hidden)
            loss += criterion(decoder_output[0], next_sentence[di])
            # Scheduled Sampling
            _, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            ni_var = Variable(torch.LongTensor([[ni]]))
            if torch.cuda.is_available():
                ni_var = ni_var.cuda()
            model_predict.append(ni_var)
            if random.random() < teacher_forcing_ratio:
                decoder_input = next_sentence[di].unsqueeze(1)
            else:
                if ni == 1: # EOS
                    break
                decoder_input = ni_var


    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), args.clip)
    torch.nn.utils.clip_grad_norm(context.parameters(), args.clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), args.clip)
    encoder_optimizer.step()
    context_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / (predict_count)

def validation(validation_data):
    validation_loss = 0
    for dialog in validation_data:
        
        context_hidden = context.init_hidden()
        
        predict_count = 0

        loss = 0

        gen_sentence = []
        for index, sentence in enumerate(dialog):
            if index == len(dialog) - 1:
                break
            decoder_input = Variable(torch.LongTensor([[0]]))
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
            encoder_hidden = encoder_hidden.view(1, 1, -1)
            context_output, context_hidden = context(encoder_hidden, context_hidden)
            next_sentence = dialog[index+1]
            for di in range(len(next_sentence)):
                predict_count += 1
                gen_sentence.append(decoder_input.data[0][0])
                decoder_output, decoder_hidden = decoder(context_hidden,\
                        decoder_input, decoder_hidden)
                loss += criterion(decoder_output[0], next_sentence[di])
                # TODO Greedy alg. now, maybe use beam search when inferencing in the future
                _, topi = decoder_output.data.topk(1)
                ni = topi[0][0]
                if ni == 1: # EOS
                    break
                decoder_input = Variable(torch.LongTensor([[ni]]))
                if torch.cuda.is_available():
                    decoder_input = decoder_input.cuda()
            # Make gen_sentence concated with a EOS and make it torch Variable
            gen_sentence.append(1)
            gen_sentence = Variable(torch.LongTensor(gen_sentence))
            if torch.cuda.is_available():
                gen_sentence = gen_sentence.cuda()

        validation_loss += (loss.data[0] / predict_count)

    return validation_loss / len(validation_data)

since = time.time()
random.shuffle(document_list)
cut = int(len(document_list) * args.validation_p)
training_data, validation_data = \
        document_list[cut:], document_list[:cut]

best_validation_score = 10000
patient = 10
model_number = 0
if args.teacher:
    teacher_forcing_ratio = 1.
else:
    teacher_forcing_ratio = 0.
for epoch in range(1, args.epochs + 1):
    training_loss = 0
    iter_since = time.time()
    try:
        for index, dialog in enumerate(training_data):
            if args.ss:
                teacher_forcing_ratio *= 0.99999
            training_loss += train(dialog)
            if (index) % 500 == 0:
                print("    @ Iter [", index + 1, "/", len(training_data),"] | avg. loss: ", training_loss / (index + 1), \
                        " | perplexity: ", math.exp(training_loss / (index + 1))," | usage ", time.time() - iter_since, " seconds | teacher_force: ", \
                        teacher_forcing_ratio)
                iter_since = time.time()
            if (index + 1) % 1000 == 0:
                validation_score_100 = validation(validation_data[:100])
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

        validation_score = validation(validation_data)
        with open(os.path.join(args.save, "val_epoch" + str(epoch) + "_" + str(model_number) + ".txt"), "w") as f:
            f.write(str(validation_score))
        print("# ", epoch, " | ", time.time() - since," seconds | validation loss: ", validation_score, " | validation perplexity: ", \
                math.exp(validation_score))
        since = time.time()
        if best_validation_score > validation_score:
            model_number += 1
            print("Saving better model number ",model_number)
            best_validation_score = validation_score
            with open(os.path.join(args.save, "encoder" + str(model_number) + ".model"), 'wb') as f:
                torch.save(encoder, f)
            with open(os.path.join(args.save, "context" + str(model_number) + ".model"), 'wb') as f:
                torch.save(context, f)
            with open(os.path.join(args.save, "decoder" + str(model_number) + ".model"), 'wb') as f:
                torch.save(decoder, f)
    except:
        print(sys.exc_info())
        model_number += 1
        print("Get stopped, saving the latest model")
        with open(os.path.join(args.save, "encoder" + str(model_number) + ".model"), 'wb') as f:
            torch.save(encoder, f)
        with open(os.path.join(args.save, "context" + str(model_number) + ".model"), 'wb') as f:
            torch.save(context, f)
        with open(os.path.join(args.save, "decoder" + str(model_number) + ".model"), 'wb') as f:
            torch.save(decoder, f)
        break

