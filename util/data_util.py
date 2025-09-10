
import torch, copy
import numpy as np
from torch_geometric.data import HeteroData
import torch.utils.data as Data

from util.temporal_interactions import Temporal_Interactions, Bi_Temporal_Interactions

def load_graph(file, device='cpu'):
    graph = torch.load(file, map_location=device)
    return graph

def load_interaction(interaction_file, device='cpu'):
    interactions = Temporal_Interactions(interaction_file, device=device)
    return interactions

def load_uk_interaction(home_path, chunk_index=0, device='cpu'):
    uk_interaction_file = f'{home_path}/chunk{chunk_index}_uk_interaction.csv'
    uk_interaction = Bi_Temporal_Interactions(file_path=uk_interaction_file, device=device)
    return uk_interaction.data

def load_target_rating(home_path, chunk_index=0, device='cpu'):
    target_rating_file = f'{home_path}/chunk{chunk_index}_target_rating.pt'
    target_rating = torch.load(target_rating_file, map_location=device) 
    return target_rating

def load_target_negative(home_path, chunk_index=0, device='cpu'):
    target_negative_file = f'{home_path}/chunk{chunk_index}_never_negative.pt'
    target_negative = torch.load(target_negative_file, map_location=device)
    return target_negative

@torch.no_grad()
def average_state_dict(dict1: dict, dict2: dict, weight: float) -> dict:
    assert 0 <= weight <= 1
    d1 = copy.deepcopy(dict1)
    d2 = copy.deepcopy(dict2)
    out = dict()
    for key in d1.keys():
        assert isinstance(d1[key], torch.Tensor)
        param1 = d1[key].detach().clone()
        assert isinstance(d2[key], torch.Tensor)
        param2 = d2[key].detach().clone()
        out[key] = (1 - weight) * param1 + weight * param2
    return out

def save_live(save_dir, chunk_index=0, GPU_usage=None, seed=123, 
         train_rating=None, valid_rating=None, test_rating=None, 
         train_metric=None, valid_metric=None, test_metric=None):
    train_rating_out_file = f'{save_dir}/chunk{chunk_index}_train_rating_{seed}.pt'
    train_metric_out_file = f'{save_dir}/chunk{chunk_index}_train_metric_never_{seed}.txt'
    valid_rating_out_file = f'{save_dir}/chunk{chunk_index}_valid_rating_{seed}.pt'
    valid_metric_out_file = f'{save_dir}/chunk{chunk_index}_valid_metric_never_{seed}.txt'
    test_rating_out_file = f'{save_dir}/chunk{chunk_index}_test_rating_{seed}.pt'
    test_metric_out_file = f'{save_dir}/test_metric_never_{seed}.txt'
    GPU_usage_out_file = f'{save_dir}/chunk{chunk_index}_GPU_usage_{seed}.txt'

    if train_rating is not None:
        torch.save(train_rating.cpu(), train_rating_out_file) 
    if train_metric is not None:
        np.savetxt(train_metric_out_file, np.array(train_metric))

    if valid_rating is not None:
        torch.save(valid_rating.cpu(), valid_rating_out_file) 
    if valid_metric is not None:
        np.savetxt(valid_metric_out_file, np.array(valid_metric))
    if test_rating is not None:
        torch.save(test_rating.cpu(), test_rating_out_file)
    if test_metric is not None:
        np.savetxt(test_metric_out_file, np.array(test_metric))
    if GPU_usage is not None:
        np.savetxt(GPU_usage_out_file, np.array(GPU_usage))
        
        
