import torch
import numpy as np

class ISMDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        three_tuple = []
        # get the features
        features = torch.tensor(self.data.iloc[idx])
        #
        three_tuple.append(features[:3])
        three_tuple.append(features[3:9])
        three_tuple.append(features[9:])

        return three_tuple
        