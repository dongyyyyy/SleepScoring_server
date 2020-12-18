import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary
from torch.utils.data import DataLoader
import torchvision.transforms as trnasforms
from pyedflib import highlevel

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import math
import time
import sys
import warnings
import datetime
import shutil


from scipy import signal
import mne
from tqdm import tnrange, tqdm

import multiprocessing
from multiprocessing import Process, Manager, Pool, Lock

from functools import partial
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device2 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device3 = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")