import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class AVTDataset(Dataset):
    def __init__(self, path, subsample=1, transforms=None):
        
        self.imgs_path = glob.glob(path+'/*')[::subsample]
        self.transforms = transforms
    
    def __len__(self):
        return len(self.imgs_path)
    
    def __getitem__(self, idx):
        sample_path = self.imgs_path[idx]
        sample = Image.open(sample_path)
        
        if self.transforms:
            sample = self.transforms(sample)
            
        return sample, 0





