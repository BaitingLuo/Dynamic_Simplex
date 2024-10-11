import numpy as np
import csv
from PIL import Image
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

class Sample_Dataset(Dataset):
    def __init__(self, data_path, n_classes, label_path, sample_length, augment_data):
        self.data_path = data_path
        self.label_path = label_path
        self.n_classes = n_classes
        self.sample_length = sample_length
        self.seq_length = 200 # Ten seconds at 20Hz
        self.augment_data = augment_data

        state_data = []
        target_data = []

        with open(self.data_path, newline='') as csvfile:
            state_data_file = list(csv.reader(csvfile, delimiter=',', quotechar='|'))
            for i, state in enumerate(state_data_file):
                state_i = []
                for s in state: state_i.append(float(s))
                state_data.append(state_i)

        with open(self.label_path, newline='') as csvfile:
            target_file = list(csv.reader(csvfile, delimiter=',', quotechar='|'))
            for i,target in enumerate(target_file):
                target_data.append(int(target[0]))        

        state_data = np.array(state_data,dtype=np.float)   
        target_data = np.array(target_data, dtype=np.uint8)  

        self.state_data = state_data
        self.target_data = target_data

    def __getitem__(self, index):
        #First, find relative index in its sequence
        index_seq = index%self.seq_length
        i_previous = index - index_seq #Number of sequences before current index
        if index_seq > self.seq_length - self.sample_length: 
            index_seq = self.seq_length - self.sample_length #Ensure not to cross over into next sequence

        index = index_seq + i_previous

        state_data = self.state_data[index:self.sample_length+index]
        target = self.target_data[index]

        #Data augmentation by adding slight noise
        if self.augment_data == True:
            noise = 0
            state_data = state_data + noise
   
        #Reshape into sequential torch input shape
        state_data = state_data.reshape((-1,3))

        state_data = torch.from_numpy(state_data).float()    
        target = torch.from_numpy(np.asarray(target)).long()

        return state_data, target  

    def __len__(self):
        return len(self.target_data)
