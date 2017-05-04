"""Hierarchical RNN implementation"""
import torch.nn as nn
from torch.autograd import Variable

class HierarchicalRNNModel(nn.Module):
    """Hierarchical RNN Building"""
    def __init__(self, rnn_type):
        super(HierarchicalRNNModel, self).__init__()
