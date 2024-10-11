import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Introspective_LSTM(nn.Module):
    def __init__(self,n_classes,size, sample_length):
        super(Introspective_LSTM, self).__init__()
        self.n_layers = 2
        self.hidden_size = size
        self.sample_length = sample_length

        self.lstm = nn.LSTM(input_size=3,
                            hidden_size=self.hidden_size,
                            num_layers=self.n_layers,
                            bidirectional=True,
                            batch_first = True)
        self.drop = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(self.hidden_size*self.sample_length*2, self.hidden_size) #bidirectional means 2 directions, hence times 2
        self.fc2 = nn.Linear(self.hidden_size,int(np.round(self.hidden_size)/2))
        self.fc3 = nn.Linear(int(np.round(self.hidden_size)/2),n_classes)

    def forward(self,input):
        batch_size = input.size(0)
        lstm_encoding,_ = self.lstm(input)
        lstm_encoding = lstm_encoding.reshape(batch_size, -1)
        lstm_encoding = self.drop(lstm_encoding)
        output = F.relu(self.fc1(lstm_encoding))
        output = self.drop(output)
        output = F.relu(self.fc2(output))
        output = self.drop(output)
        output = self.fc3(output)

        return output