from include.header import *


class BiLSTM_onePerson(nn.Module):
    def __init__(self, input_size=128, class_num=5, hidden_dim=256, num_layers=1, use_gpu=True,sequence_length=10):
        super(BiLSTM_onePerson, self).__init__()
        self.input_size = input_size
        self.num_directions = 2
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        
        self.bi_lstm1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_dim, num_layers=num_layers,
                                batch_first=True, bidirectional=True)
        print(self.bi_lstm1)

        self.hidden2label = nn.Linear(hidden_dim * 2, class_num)
        self.shortcut = nn.Linear(self.input_size, hidden_dim * 2)

    def init_hidden1(self, batch_size):
        # first = h(hidden) / second = c(cell)
        if self.use_gpu:
            return (
            Variable(torch.zeros(self.num_directions * self.num_layers, batch_size, self.hidden_dim).cuda()),  # hidden
            Variable(torch.zeros(self.num_directions * self.num_layers, batch_size, self.hidden_dim).cuda()))  # cell
        else:
            return (Variable(torch.zeros(self.num_directions * self.num_layers, batch_size, self.hidden_dim)),  # hidden
                    Variable(torch.zeros(self.num_directions * self.num_layers, batch_size, self.hidden_dim)))  # cell

    def forward(self, input):
        shortcut_output = input
        
        input = input.unsqueeze(0) # batch_size =1, cnn에서의 batch = sequence length
        # [1, sequence_length, vector_size]

        shortcut_output = self.shortcut(shortcut_output)
        self.hidden1 = self.init_hidden1(input.size(0))
        output, self.hidden1 = self.bi_lstm1(input, self.hidden1)

        output = output + shortcut_output
        y = self.hidden2label(output) # classification
        y = y.squeeze(0)#[1,sequence_length, class_num] -> [sequence_length(batch size),class_num]
        return y

class BiLSTM_sequence(nn.Module):
    def __init__(self, input_size=128, class_num=5, hidden_dim=256, num_layers=1, use_gpu=True,sequence_length=10):
        super(BiLSTM_sequence, self).__init__()
        self.input_size = input_size
        self.num_directions = 2
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.sequence_length = sequence_length
        self.bi_lstm1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_dim, num_layers=num_layers,
                                batch_first=True, bidirectional=True)
        print(self.bi_lstm1)

        self.hidden2label = nn.Linear(hidden_dim * 2, class_num)
        self.shortcut = nn.Linear(self.input_size, hidden_dim * 2)

    def init_hidden1(self, batch_size):
        # first = h(hidden) / second = c(cell)
        if self.use_gpu:
            return (
            Variable(torch.zeros(self.num_directions * self.num_layers, batch_size, self.hidden_dim).cuda()),  # hidden
            Variable(torch.zeros(self.num_directions * self.num_layers, batch_size, self.hidden_dim).cuda()))  # cell
        else:
            return (Variable(torch.zeros(self.num_directions * self.num_layers, batch_size, self.hidden_dim)),  # hidden
                    Variable(torch.zeros(self.num_directions * self.num_layers, batch_size, self.hidden_dim)))  # cell

    def forward(self, input):
        # print('input shape : ',input.shape)
        shortcut_output = input
        batch_size = input.size(0)//self.sequence_length
        # print(f'batch_size : {batch_size} / input.size(-1) : {input.size(-1)} / sequence_length : {self.sequence_length}')
        input = input.reshape(batch_size,self.sequence_length,input.size(-1))
        
        # print('shortcut shape : ',shortcut_output.shape)
        shortcut_output = self.shortcut(shortcut_output)
        self.hidden1 = self.init_hidden1(input.size(0))
        output, self.hidden1 = self.bi_lstm1(input, self.hidden1)
        # print('output shape : ',output.shape)

        # shortcut Add
        # print('output shape : ',output.shape)
        # print('shortcut_output shape : ',shortcut_output)
        shortcut_output = shortcut_output.reshape(batch_size,self.sequence_length,self.hidden_dim*2)
        output = output + shortcut_output
        y = self.hidden2label(output)
        y = y.squeeze(0)
        return y


class lstm(nn.Module):
    def __init__(self, in_channel=1, flat=3456 * 2, num_layers=1, hidden_dim=512, num_classes=5,sequence_length=10):
        super(lstm, self).__init__()
        self.classification = BiLSTM_sequence(class_num=num_classes, input_size=flat,
                                                            hidden_dim=hidden_dim, num_layers=num_layers,sequence_length=sequence_length)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, input):
        out = self.dropout(input)
        out = self.classification(out)

        return out

class lstm_onePerson(nn.Module):
    def __init__(self,flat = 3456*2, num_layers=1,hidden_dim=512,num_classes=5):
        super(lstm_onePerson,self).__init__()
        self.classification = BiLSTM_onePerson(class_num=num_classes,input_size=flat,
                                                hidden_dim=hidden_dim,num_layers=num_layers)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, input):
        out = self.dropout(input)
        out = self.classification(out)

        return out