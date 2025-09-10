import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class Temporal_Interactions(Dataset):
    def __init__(self, file_path, device='cpu') -> None:
        interaction_df = pd.read_csv(file_path, sep='\t', dtype=np.float32)
        interaction_df['2'] = interaction_df['2'] - interaction_df['2'].iloc[0]
        self.data = torch.from_numpy(interaction_df.values).to(device)
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        sample = self.data[index, :]
        return sample
    
class Bi_Temporal_Interactions(Dataset):
    def __init__(self, file_path, device='cpu') -> None:
        interaction_df = pd.read_csv(file_path, sep='\t', dtype=np.float32)
        # interaction_df['2'] = interaction_df['2']/3600
        self.data = torch.from_numpy(interaction_df.values).to(device)
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        sample = self.data[index, :]
        return sample