import torch
from torch import Tensor
    
class LastNeighborLoader:
    def __init__(self, num_nodes: dict, size: int, device='cpu'):
        self.size = size
        self.device = device
        self.neighbors = torch.full(size=(num_nodes['user'], num_nodes['concept']), fill_value=-1, dtype=torch.float, device=self.device)
        self.neighbors_copy = self.neighbors.clone()

    def __call__(self, node_id: Tensor, node_type:str) -> Tensor:
        if node_type == 'user':
            neighbors_value = self.neighbors[node_id]
        else:
            neighbors_value = self.neighbors.t()[node_id]
        neighbors_value, neighbors = torch.sort(neighbors_value, descending=True)
        top_k_neighbors = neighbors[:, :self.size]
        top_k_values = neighbors_value[:, :self.size]
        # top_k_values, top_k_neighbors = torch.topk(neighbors_value, self.size, dim=1, largest=True, sorted=True)
        top_k_neighbors = torch.where(top_k_values!=-1, top_k_neighbors, -1)
        return top_k_neighbors.long()

    def insert(self, edges: Tensor, current_times: Tensor):
        src_ids = edges[:, 0]
        dst_ids = edges[:, 1]
        self.neighbors[src_ids, dst_ids] = current_times

    def reset_params(self):
        self.neighbors = self.neighbors_copy.clone()

    def update_params(self, neighbors:Tensor):
        self.neighbors_copy = neighbors.clone()
    
    def get_params(self):
        return self.neighbors.clone()
