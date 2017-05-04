"""An-Automated-Traditional-Chinese-Dialogue-Generating-System Main file"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

parser = argparse.ArgumentParser(description=\
        'Pytorch Traditional Chinese Dialouge Generating System builded on Hierarchical RNN.')

