import torch
from torch_geometric.data import HeteroData

class GraphData():
    def __init__(self, node_nums:dict, node_dims:int, hidden_dims:int, has_pref=False, has_knowl=False, device='cpu') -> None:
        self.node_nums = node_nums
        self.node_dims = node_dims
        self.hidden_dims = hidden_dims
        self.has_knowl = has_knowl
        self.has_pref = has_pref
        self.device = device
        self.graph = HeteroData()

    def generate_graph(self):
        for node_type in self.node_nums.keys():
            node_num = self.node_nums[node_type]
            if self.has_pref:
                self.graph[node_type].preference = torch.randn(size=(node_num, self.node_dims), device=self.device)
                self.graph[node_type].pref_cell = torch.randn(size=(node_num, self.hidden_dims), device=self.device)
                self.graph[node_type].pref_hidden = torch.randn(size=(node_num, self.hidden_dims), device=self.device)
            if self.has_knowl:
                self.graph[node_type].knowledge = torch.randn(size=(node_num, self.node_dims), device=self.device)
                self.graph[node_type].knowl_hidden = torch.randn(size=(node_num, self.hidden_dims), device=self.device)
                self.graph[node_type].knowl_cell = torch.randn(size=(node_num, self.hidden_dims), device=self.device)

    def get_graph(self):
        return self.graph.detach().clone()
    
    def set_graph(self, graph:HeteroData):
        self.graph = graph