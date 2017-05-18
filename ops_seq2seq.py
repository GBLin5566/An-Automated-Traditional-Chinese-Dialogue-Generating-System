# -*- coding: utf-8 -*-
'''Operations for seq2seq model'''
from builtins import range

import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import check_cuda_for_var

def train(my_lang, criterion, teacher_forcing_ratio, \
        training_data, encoder, decoder,\
        encoder_optimizer, decoder_optimizer, max_length):
    total_loss = 0
    predict_num = 0
    for index, sentence in enumerate(training_data):
        if index == len(training_data) - 1:
            break
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss = 0

        encoder_hidden = encoder.init_hidden()
        encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
        decoder_input = Variable(torch.LongTensor([[my_lang.word2index["SOS"]]]))
        encoder_outputs = check_cuda_for_var(encoder_outputs)
        decoder_input = check_cuda_for_var(decoder_input)

        for ei in range(len(sentence)):
            encoder_output, encoder_hidden = encoder(sentence[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0][0]

        decoder_hidden = encoder_hidden

        next_sentence = training_data[index+1]
        if random.random() < teacher_forcing_ratio:
            for di in range(len(next_sentence)):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, \
                        encoder_outputs)
                loss += criterion(decoder_output[0], next_sentence[di])
                predict_num += 1
                decoder_input = next_sentence[di]
        else:
            for di in range(len(next_sentence)):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, \
                        encoder_output, encoder_outputs)
                loss += criterion(decoder_output[0], next_sentence[di])
                predict_num += 1
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]

                decoder_input = Variable(torch.LongTensor([[ni]]))
                decoder_input = check_cuda_for_var(decoder_input)
        total_loss += loss
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    
    return total_loss.data[0] / predict_num

def validate(my_lang, criterion, validation_data, encoder, decoder):
    total_loss = 0
    predict_num = 0
    for dialog in validation_data:
        for index, sentence in enumerate(dialog):
            if index == len(training_data) - 1:
                break
            loss = 0

            encoder_hidden = encoder.init_hidden()
            encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
            decoder_input = Variable(torch.LongTensor([[my_lang.word2index["SOS"]]]))
            encoder_outputs = check_cuda_for_var(encoder_outputs)
            decoder_input = check_cuda_for_var(decoder_input)

            for ei in range(len(sentence)):
                encoder_output, encoder_hidden = encoder(sentence[ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output[0][0]

            decoder_hidden = encoder_hidden

            next_sentence = training_data[index+1]
            for di in range(len(next_sentence)):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, \
                        encoder_output, encoder_outputs)
                loss += criterion(decoder_output[0], next_sentence[di])
                predict_num += 1
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]

                decoder_input = Variable(torch.LongTensor([[ni]]))
                decoder_input = check_cuda_for_var(decoder_input)
            total_loss += loss
    return total_loss.data[0] / predict_num
def sample(my_lang, dialog, encoder, decoder):
    pass
