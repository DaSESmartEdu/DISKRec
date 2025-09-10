import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from model.dgnn import DGNN
from model.rate_layer import RateLayer

class DISKRec(nn.Module):
    def __init__(self, node_nums, args, device='cpu'):
        super(DISKRec, self).__init__()
        self.name = 'DISKRec'
        self.device = device
        self.args = args
        self.state_mode = args.state_mode
        self.node_nums = node_nums
        
        self.dgnn = DGNN(node_nums, args=args, device=device)
        
        if args.act == 'tanh': self.act = nn.Tanh()
        elif args.act == 'relu': self.act = nn.ReLU()
        else: self.act = nn.Sigmoid()

        if self.state_mode == 'both': self.init_fusion_func()
        
        self.rate_layer = RateLayer(node_nums['user'], node_nums['concept'], args.latent_dim, args.node_dims)
        self.rate_loss = nn.MSELoss(reduction='mean')
    
    def init_fusion_func(self):
        if self.args.fusion == 'linear':
            self.fusion = nn.Sequential(nn.Linear(2*self.args.node_dims, self.args.node_dims), self.act)
            # nn.init.xavier_uniform_(self.fusion[0].weight)
            # nn.init.zeros_(self.fusion[0].bias)
        elif self.args.fusion == 'gating':
            self.fusion = nn.Sequential(nn.Linear(2*self.args.node_dims, self.args.node_dims), self.act)
            self.gating = nn.Sequential(nn.Linear(2*self.args.node_dims, self.args.node_dims), nn.Sigmoid())

    def fusion_func(self, n_ps, n_ks):
        if self.args.fusion == 'linear':
            n_F = self.fusion(torch.cat([n_ps, n_ks], dim=-1))
        elif self.args.fusion == 'gating':
            n_F = self.act(self.fusion(torch.cat([n_ps, n_ks], dim=-1)))
            gate = self.gating(torch.cat([n_ps, n_ks], dim=-1))
            n_F = n_F * gate
        return n_F

    def forward(self, intpu_graph:HeteroData, chunk_data:Tensor, target_rating=None): 
        out_graph = self.dgnn(intpu_graph, chunk_data)

        if self.state_mode in ['pref']:
            user = out_graph['user'].preference
            item = out_graph['concept'].preference
        elif self.state_mode == 'knowl':
            user = out_graph['user'].knowledge
            item = out_graph['concept'].knowledge
        else:
            user_pref = out_graph['user'].preference
            item_pref = out_graph['concept'].preference
            user_knowl = out_graph['user'].knowledge
            item_knowl = out_graph['concept'].knowledge
            user = self.fusion_func(user_pref, user_knowl)
            item = self.fusion_func(item_pref, item_knowl)
        
        pred_rating = self.rate_layer(user, item)
        loss = self.rate_loss(pred_rating, target_rating)
                
        res = {
            'out_graph': out_graph.detach().clone(),
            'rating': pred_rating.detach().clone(),
            'loss': loss
        }
        return res
