# -*- coding: utf-8 -*-
'''Operations for h-rnn model'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import check_cuda_for_var

def train(my_lang, criterion, teacher_forcing_ratio, \
        training_data, encoder, decoder,\
        encoder_optimizer, decoder_optimizer):
    pass

def validate(my_lang, criterion, teacher_forcing_ratio, \
        validation_data, encoder, decoder,\
        encoder_optimizer, decoder_optimizer):
    pass

def sample(my_lang, dialog, encoder, decoder):
    pass
