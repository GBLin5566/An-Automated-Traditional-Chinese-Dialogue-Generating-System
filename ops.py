# -*- coding: utf-8 -*-
'''Operations for h-rnn model'''
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import check_cuda_for_var

def train(my_lang, criterion, teacher_forcing_ratio, \
        training_data, encoder, context, decoder,\
        encoder_optimizer, context_optimizer, decoder_optimizer):
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
        decoder_input = Variable(torch.LongTensor([[my_lang.word2index["SOS"]]]))
        decoder_input = check_cuda_for_var(decoder_input)
        encoder_hidden = encoder.init_hidden()
        for ei in range(len(sentence)):
            if ei > len(model_predict) - 1 or random.random() < teacher_forcing_ratio:
                _, encoder_hidden = encoder(sentence[ei], encoder_hidden)
            else:
                _, encoder_hidden = encoder(model_predict[ei], encoder_hidden)
        # Assign last encoder's hidden to decoder
        decoder_hidden = encoder_hidden
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
                decoder_input = ni_var

    loss.backward()
    encoder_optimizer.step()
    context_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / (predict_count)

def validate(my_lang, criterion, teacher_forcing_ratio, \
        validation_data, encoder, context, decoder,\
        encoder_optimizer, context_optimizer, decoder_optimizer):
    validation_loss = 0
    for dialog in validation_data:

        context_hidden = context.init_hidden()

        predict_count = 0

        loss = 0

        gen_sentence = []
        for index, sentence in enumerate(dialog):
            if index == len(dialog) - 1:
                break
            decoder_input = Variable(torch.LongTensor([[my_lang.word2index["SOS"]]]))
            decoder_input = check_cuda_for_var(decoder_input)
            encoder_hidden = encoder.init_hidden()
            if len(gen_sentence) > 0:
                for ei in range(len(gen_sentence)):
                    _, encoder_hidden = encoder(gen_sentence[ei], encoder_hidden)
                # Clean generated sentence list
                gen_sentence = []
            else:
                for ei in range(len(sentence)):
                    _, encoder_hidden = encoder(sentence[ei], encoder_hidden)
            decoder_hidden = encoder_hidden
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
                #if ni == 1: # EOS
                #    break
                decoder_input = Variable(torch.LongTensor([[ni]]))
                if torch.cuda.is_available():
                    decoder_input = decoder_input.cuda()
            # Make gen_sentence concated with a EOS and make it torch Variable
            gen_sentence.append(my_lang.word2index["EOS"])
            gen_sentence = Variable(torch.LongTensor(gen_sentence))
            if torch.cuda.is_available():
                gen_sentence = gen_sentence.cuda()
        
        validation_loss += (loss.data[0] / predict_count)
        
    return validation_loss / len(validation_data)

def sample(my_lang, dialog, encoder, context, decoder):
    print("Golden ->")
    for sentence in dialog:
        string = ' '.join([my_lang.index2word[word.data[0]] for word in sentence])
        print(string)
    print("Predict ->")
    gen_sentence = []
    context_hidden = context.init_hidden()
    for index, sentence in enumerate(dialog):
        if index == len(dialog) - 1:
            break
        decoder_input = Variable(torch.LongTensor([[my_lang.word2index["SOS"]]]))
        decoder_input = check_cuda_for_var(decoder_input)
        encoder_hidden = encoder.init_hidden()
        if len(gen_sentence) > 0:
            for ei in range(len(gen_sentence)):
                _, encoder_hidden = encoder(gen_sentence[ei], encoder_hidden)
            # Clean generated sentence list
            gen_sentence = []
        else:
            for ei in range(len(sentence)):
                _, encoder_hidden = encoder(sentence[ei], encoder_hidden)
        decoder_hidden = encoder_hidden
        context_output, context_hidden = context(encoder_hidden, context_hidden)
        next_sentence = dialog[index+1]
        for di in range(len(next_sentence)):
            gen_sentence.append(decoder_input.data[0][0])
            decoder_output, decoder_hidden = decoder(context_hidden,\
                    decoder_input, decoder_hidden)
            _, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            decoder_input = Variable(torch.LongTensor([[ni]]))
            if torch.cuda.is_available():
                decoder_input = decoder_input.cuda()
        # Make gen_sentence concated with a EOS and make it torch Variable
        gen_sentence.append(my_lang.word2index["EOS"])
        gen_sentence = Variable(torch.LongTensor(gen_sentence))
        if torch.cuda.is_available():
            gen_sentence = gen_sentence.cuda()
        string = ' '.join([my_lang.index2word[word.data[0]] for word in gen_sentence])
        print(string)
