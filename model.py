"""Hierarchical RNN implementation"""
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.cuda
from torch.autograd import Variable

class EncoderRNN(nn.Module):
    """Encoder RNN Building"""
    def __init__(self, input_size, hidden_size, n_layers, dropout):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)

        self.cuda = torch.cuda.is_available()
        self.init_weight()

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if self.cuda:
            hidden = hidden.cuda()
        return hidden

    def init_weight(self):
        init.orthogonal(self.embedding.weight.data)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden


class ContextRNN(nn.Module):
    """Context RNN Building"""
    def __init__(self, encoder_hidden_size, hidden_size, n_layers, dropout):
        super(ContextRNN, self).__init__()
        
        self.encoder_hidden_size = encoder_hidden_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.mapping = nn.Linear(encoder_hidden_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        
        self.cuda = torch.cuda.is_available()
        self.init_weight()

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if self.cuda:
            hidden = hidden.cuda()
        return hidden

    def init_weight(self):
        init.orthogonal(self.mapping.weight.data)

    def forward(self, input, hidden):
        mapping = self.mapping(input)
        output, hidden = self.gru(mapping, hidden)
        return output, hidden


class DecoderRNN(nn.Module):
    """Decoder RNN Building"""
    def __init__(self, hidden_size, output_size, n_layers, dropout):
        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.out = nn.Linear(hidden_size, output_size)

        self.cuda = torch.cuda.is_available()
        self.init_weight()

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if self.cuda:
            hidden = hidden.cuda()
        return hidden

    def init_weight(self):
        init.orthogonal(self.out.weight.data)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        output = F.log_softmax(self.out(output))
        return output, hidden
