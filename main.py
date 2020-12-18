from include.header import *
from train.cnn.train_cnn import *
from train.transformer.train_transformer import *
from train.rnn.train_rnn import *
from train.rnn.train_rnn_cuda1 import *
from train.rnn.train_rnn_onePerson import *
from train.cnn.train_cnn_seoul import *

use_cudaNum = 1
sequence_length = 50
window_size = 50
hidden_dim = 512
num_layers = 2
if __name__=='__main__':
    # training_rnn_dataloader_onePerson(use_cudaNum=use_cudaNum,sequence_length=sequence_length,window_size=window_size,num_layers=num_layers,hidden_dim=hidden_dim)
    training_cnn_dataloader_seoul()
    # training_cnn_dataloader()
    # training_detr_dataloader()