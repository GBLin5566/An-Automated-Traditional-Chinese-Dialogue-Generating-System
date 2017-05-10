"""An-Automated-Traditional-Chinese-Dialogue-Generating-System Main file"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time

import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable

import model

learning_rate = 0.01
encoder = model.EncoderRNN(10, 1000, 2, 1)
context = model.ContextRNN(2*1000, 2000, 2, 1)
decoder = model.DecoderRNN(2*2000, 500, 10, 2, 1)
if torch.cuda.is_available():
    encoder = encoder.cuda()
    context = context.cuda()
    decoder = decoder.cuda()
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
context_optimizer = optim.Adam(context.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

parser = argparse.ArgumentParser(description=\
        'Pytorch Traditional Chinese Dialouge Generating System builded on Hierarchical RNN.')

def check_cuda_for_var(var):
    if torch.cuda.is_available():
        return var.cuda()
    else:
        return var

def train():
    # Zero gradients
    encoder_optimizer.zero_grad()
    context_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = Variable(torch.FloatTensor(1))
    nn.init.constant(loss, 0)
    loss = check_cuda_for_var(loss)

    context_hidden = context.init_hidden()

    talk_history = [[0,3,2,5,1,5,9],[0,8,6,4,9],[0,4,2,5,1,2,9],[0,3,6,2,6,8,6,9],[0,3,1,1,1,9],[0,4,5,3,9]]
    predict_count = 0

    for index, sentence in enumerate(talk_history):
        if index == len(talk_history) - 1:
            break
        decoder_input = Variable(torch.LongTensor([[0]]))
        decoder_input = check_cuda_for_var(decoder_input)
        encoder_hidden = encoder.init_hidden()
        decoder_hidden = decoder.init_hidden()
        sentence_variable = Variable(torch.LongTensor(sentence))
        sentence_variable = check_cuda_for_var(sentence_variable)
        for ei in range(len(sentence)):
            _, encoder_hidden = encoder(sentence_variable[ei], encoder_hidden)
        encoder_hidden = encoder_hidden.view(1, 1, -1)
        context_output, context_hidden = context(encoder_hidden, context_hidden)
        next_sentence_variable = Variable(torch.LongTensor(talk_history[index+1]))
        next_sentence_variable = check_cuda_for_var(next_sentence_variable)
        for di in range(len(talk_history[index+1])):
            decoder_output, decoder_hidden = decoder(context_hidden,\
                    decoder_input, decoder_hidden)
            loss += criterion(decoder_output[0], next_sentence_variable[di])
            decoder_input = next_sentence_variable[di].unsqueeze(1)
            predict_count += 1

    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), 5.0)
    torch.nn.utils.clip_grad_norm(context.parameters(), 5.0)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), 5.0)
    encoder_optimizer.step()
    context_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / (predict_count)

since = time.time()
for epoch in range(1, 7001):
    training_loss = train()
    if epoch % 10  == 0:
        print("# ", epoch, " | ", time.time() - since," seconds | loss: ", training_loss)
        since = time.time()
