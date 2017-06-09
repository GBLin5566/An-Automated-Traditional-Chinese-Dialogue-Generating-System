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
    # Training mode
    encoder.train()
    decoder.train()
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
                        encoder_outputs)
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

def validate(my_lang, criterion, validation_data, encoder, decoder, max_length):
    total_loss = 0
    predict_num = 0
    # Eval mode
    encoder.eval()
    decoder.eval()
    for counter, dialog in enumerate(validation_data):
        if counter == len(validation_data) - 1:
            sample(my_lang, dialog, encoder, decoder, max_length)
        for index, sentence in enumerate(dialog):
            if index == len(dialog) - 1:
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

            next_sentence = dialog[index+1]
            for di in range(len(next_sentence)):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, \
                        encoder_outputs)
                loss += criterion(decoder_output[0], next_sentence[di])
                predict_num += 1
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]

                decoder_input = Variable(torch.LongTensor([[ni]]))
                decoder_input = check_cuda_for_var(decoder_input)
            if isinstance(loss, float):
                total_loss += loss
            else:
                total_loss += loss.data[0]
    return total_loss / predict_num
def sample(my_lang, dialog, encoder, decoder, max_length):
    # Eval mode
    encoder.eval()
    decoder.eval()
    print("Golden ->")
    for sentence in dialog:
        string = ' '.join([my_lang.index2word[word.data[0]] for word in sentence])
        print(string)
    print("Predict ->")
    gen_sentence = []
    for index, sentence in enumerate(dialog):
        if index == len(dialog) - 1:
            break
        encoder_hidden = encoder.init_hidden()
        encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
        decoder_input = Variable(torch.LongTensor([[my_lang.word2index["SOS"]]]))
        encoder_outputs = check_cuda_for_var(encoder_outputs)
        decoder_input = check_cuda_for_var(decoder_input)

        if len(gen_sentence) > 0:
            for ei in range(len(gen_sentence)):
                encoder_output, encoder_hidden = encoder(gen_sentence[ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output[0][0]
            gen_sentence = []
        else:
            for ei in range(len(sentence)):
                encoder_output, encoder_hidden = encoder(sentence[ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output[0][0]

        decoder_hidden = encoder_hidden

        next_sentence = dialog[index+1]
        for di in range(len(next_sentence)):
            gen_sentence.append(decoder_input.data[0][0])
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, \
                    encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = check_cuda_for_var(decoder_input)
        gen_sentence.append(my_lang.word2index["EOS"])
        gen_sentence = Variable(torch.LongTensor(gen_sentence))
        gen_sentence = check_cuda_for_var(gen_sentence)
        string = ' '.join([my_lang.index2word[word.data[0]] for word in gen_sentence])
        print(string)
