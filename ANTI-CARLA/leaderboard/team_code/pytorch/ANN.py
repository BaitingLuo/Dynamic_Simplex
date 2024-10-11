import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, x_d, y_d):
        super().__init__()
        self.board_x, self.board_y = x_d, y_d


        self.hidden_1 = nn.Linear(self.board_x, 512)
        self.hidden_2 = nn.Linear(512, 256)
        self.hidden_3 = nn.Linear(256, 128)
        self.hidden_4 = nn.Linear(128, 64)
        self.hidden_5 = nn.Linear(64, 32)
        self.output1 = nn.Linear(32, 1)

        # Define sigmoid activation and softmax output
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = x.view(-1,1,self.board_y, self.board_x)
        x = self.hidden_1(x)
        x = self.relu(x)
        x = self.hidden_2(x)
        x = self.relu(x)
        x = self.hidden_3(x)
        x = self.relu(x)
        x = self.hidden_4(x)
        x = self.relu(x)
        x = self.hidden_5(x)
        x = self.relu(x)
        x = x.view(-1, 32)
        pi = self.output1(x)
        return nn.functional.sigmoid(pi)
        #return nn.functional.relu(pi), nn.functional.relu(v)
        #return nn.functional.log_softmax(pi, dim =1), torch.tanh(v)