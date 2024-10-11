#!/bin/python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Controller_Selector(nn.Module):
    def __init__(self,n_classes,size):
        super(Controller_Selector, self).__init__()
        self.n_states = 14 # 8 if not one-hot encoded, 14 if it is
        self.fc1 = nn.Linear(self.n_states, size)
        self.fc2 = nn.Linear(size, size)
        self.drop = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(size,np.round(size/2).astype(int))
        self.fc5 = nn.Linear(np.round(size/2).astype(int),n_classes)

    def forward(self,x):
        x1 = F.relu(self.fc1(x))
        x1 = self.drop(x1)
        x2 = F.relu(self.fc2(x1))
        x3 = self.drop(x2)
        x4 = F.relu(self.fc4(x3))
        x4 = self.drop(x4)
        output = F.relu(self.fc5(x4))

        return output
