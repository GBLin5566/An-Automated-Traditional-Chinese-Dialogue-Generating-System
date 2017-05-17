"""Hierarchical RNN implementation"""
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.cuda
from torch.autograd import Variable

# TODO
# Init. GRU with orthogonal initializer.
class EncoderRNN(nn.Module):
    """Encoder RNN Building"""
    def __init__(self, input_size, hidden_size, n_layers, dropout):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)

        self.is_cuda = torch.cuda.is_available()
        #self.init_weight()

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if self.is_cuda:
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

        self.gru = nn.GRU(encoder_hidden_size, hidden_size, n_layers, dropout=dropout)
        
        self.is_cuda = torch.cuda.is_available()

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if self.is_cuda:
            hidden = hidden.cuda()
        return hidden

    def forward(self, input, hidden):
        input = input.view(1, 1, -1)
        output, hidden = self.gru(input, hidden)
        return output, hidden


class DecoderRNN(nn.Module):
    """Decoder RNN Building"""
    def __init__(self, context_output_size, hidden_size, output_size, n_layers, dropout):
        super(DecoderRNN, self).__init__()

        self.context_output_size = context_output_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        
        self.out = nn.Linear(hidden_size, output_size)
        self.gru = nn.GRU(context_output_size + hidden_size, hidden_size, n_layers, dropout=dropout)

        self.is_cuda = torch.cuda.is_available()
        #self.init_weight()

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if self.is_cuda:
            hidden = hidden.cuda()
        return hidden

    def init_weight(self):
        init.orthogonal(self.out.weight.data)

    def forward(self, context_output, input, hidden):
        context_output = context_output.view(1, 1, -1)
        input_cat = torch.cat([context_output, self.embedding(input)], 2)
        output, hidden = self.gru(input_cat, hidden)
        output = F.log_softmax(self.out(output[0]))
        return output, hidden
