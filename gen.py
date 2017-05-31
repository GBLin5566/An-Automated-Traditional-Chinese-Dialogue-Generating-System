# -*- coding: utf-8 -*-
"""Generator for model"""
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
from ops import train, validate, sample
import utils
from utils import check_cuda_for_var, check_directory

parser = argparse.ArgumentParser(description=\
        "Generator for HRNN/Seq2seq")
parser.add_argument('--data', type=str,
        help="location of the data corpus(json file)")
parser.add_argument('--model_dir', type=str,
        help="location of the model's directory")
parser.add_argument('--type', type=str,
        help="generate dialog with hrnn/seq2seq model")
args = parser.parse_args()

if args.type != "hrnn" or args.type != "seq2seq":
    raise ValueError("args.type should be hrnn or seq2seq, but got %s" % (args.type))
