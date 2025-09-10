import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from tqdm import tqdm
from torch_scatter import scatter_max, segment_csr

from model.combiner import Combiner
from model.edge_message import EdgeMessage
from model.iterative_updater import node_LSTM
from model.decayer import Decayer
from model.neighbor_algorithm import LastNeighborLoader

class NeighborAttention(nn.Module):
    def __init__(self):
        super(NeighborAttention,self).__init__()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, node1, node2):
        output = torch.matmul(node1, node2.transpose(1,2)).sum(dim=2)
        score = self.softmax(output)
        return score

class DGNN(nn.Module):
    def __init__(self, node_nums:dict, args, device='cpu'):
        super(DGNN,self).__init__()
        self.node_nums = node_nums
        self.args = args
        self.node_dims = args.node_dims
        self.message_dims = args.message_dims
        self.hidden_dims = args.hidden_dims
        self.neighbors = args.neighbors
        self.decay = args.decay
        self.cross = args.cross
        self.state_mode = args.state_mode
        self.nei_algo = args.nei_algo
        self.batch_size = args.batch_size
        self.msg_agg = args.msg_agg
        self.device = device
        
        if args.act == 'tanh': self.act = nn.Tanh()
        elif args.act == 'relu': self.act = nn.ReLU()
        else: self.act = nn.Sigmoid()

        self.decayer = Decayer(args.w, args.decay_method)
        self.neighbor_loader = LastNeighborLoader(self.node_nums, size=self.neighbors, device=self.device)

        self.node_timestamp = {}
        self.node_timestamp_copy = {}

        if self.state_mode in ['both', 'pref']:
            self.pref_message = nn.ModuleDict()
            self.pref_updater = nn.ModuleDict()
            self.pref_nei_att = nn.ModuleDict()
            self.node_timestamp['pref'] = {}
            self.node_timestamp_copy['pref'] = {}
            for node_type in self.node_nums.keys():
                self.pref_message[node_type] = EdgeMessage(self.node_dims, self.message_dims, act=self.act)
                self.pref_updater[node_type] = node_LSTM(self.message_dims, self.hidden_dims, self.decay, self.cross)
                self.pref_nei_att[node_type] = NeighborAttention()
                self.node_timestamp['pref'][node_type] = torch.zeros(size=(self.node_nums[node_type], 1), dtype=torch.float, device=device)
                self.node_timestamp_copy['pref'][node_type] = self.node_timestamp['pref'][node_type].clone()
            self.pref_combiner = Combiner(self.hidden_dims, self.node_dims, act=self.act)
            if args.trans_message: 
                self.trans_pref_message = nn.Linear(self.message_dims, self.hidden_dims)
                nn.init.xavier_normal_(self.trans_pref_message.weight)
                nn.init.zeros_(self.trans_pref_message.bias)

        if self.state_mode in ['both', 'knowl']:
            self.knowl_message = nn.ModuleDict()
            self.knowl_updater = nn.ModuleDict()
            self.knowl_nei_att = nn.ModuleDict()
            self.node_timestamp['knowl'] = {}
            self.node_timestamp_copy['knowl'] = {}
            for node_type in self.node_nums.keys():
                nei_type = 'user' if node_type == 'concept' else 'concept'
                self.knowl_message[node_type] = EdgeMessage(self.node_dims, self.message_dims, 2*self.node_nums[nei_type], act=self.act)
                self.knowl_updater[node_type] = node_LSTM(self.message_dims, self.hidden_dims, self.decay, self.cross)
                self.knowl_nei_att[node_type] = NeighborAttention()
                self.node_timestamp['knowl'][node_type] = torch.zeros(size=(self.node_nums[node_type], 1), dtype=torch.float, device=device)
                self.node_timestamp_copy['knowl'][node_type] = self.node_timestamp['knowl'][node_type].clone()
            self.knowl_combiner = Combiner(self.hidden_dims, self.node_dims, act=self.act)
            if args.trans_message: 
                self.trans_knowl_message = nn.Linear(self.message_dims, self.hidden_dims)
                nn.init.xavier_normal_(self.trans_knowl_message.weight)
                nn.init.zeros_(self.trans_knowl_message.bias)

    def reset_params(self):
        for status_type in self.node_timestamp_copy.keys():
            for node_type, tensor in self.node_timestamp_copy[status_type].items():
                self.node_timestamp[status_type][node_type] = tensor.clone()
        self.neighbor_loader.reset_params()

    def update_params(self, node_timestamp:dict, neighbors:torch.Tensor):
        for status_type in node_timestamp.keys():
            for node_type, tensor in node_timestamp[status_type].items():
                self.node_timestamp[status_type][node_type] = tensor.clone()
        self.neighbor_loader.update_params(neighbors)
    
    def get_params(self):
        node_timestamp = {}
        for status_type in self.node_timestamp.keys():
            node_timestamp[status_type] = {}
            for node_type, tensor in self.node_timestamp[status_type].items():
                node_timestamp[status_type][node_type] = tensor.clone()
        neighbors = self.neighbor_loader.get_params()
        return node_timestamp, neighbors

    def merged_by_segment_csr(self, node_ids, message, timestamps):
        sorted_node_ids, sorted_node_indices = torch.sort(node_ids)
        sorted_message = message[sorted_node_indices]
        sorted_timestamps = timestamps[sorted_node_indices]
        unique_node_ids, unique_source_counts = torch.unique_consecutive(sorted_node_ids, return_counts=True)
        node_segment_indices = torch.cumsum(unique_source_counts, dim=0) - unique_source_counts
        node_segment_indices = torch.cat((node_segment_indices, torch.tensor([len(sorted_node_ids)], device=self.device)))
        merged_message = segment_csr(sorted_message, node_segment_indices, reduce=self.msg_agg)
        merged_current_t = segment_csr(sorted_timestamps, node_segment_indices, reduce='max')
        return unique_node_ids, merged_message, merged_current_t
    
    def knowl_encode(self, source_ids, target_ids, is_correct):
        user_encode = torch.zeros(size=(source_ids.shape[0], 2*self.node_nums['concept']), device=self.device)
        user_encode[:, :self.node_nums['concept']][is_correct == 1, target_ids[is_correct == 1]] = 1
        user_encode[:, self.node_nums['concept']:][is_correct == 0, target_ids[is_correct == 0]] = 1

        item_encode = torch.zeros(size=(target_ids.shape[0], 2*self.node_nums['user']), device=self.device)
        item_encode[:, :self.node_nums['user']][is_correct == 1, source_ids[is_correct == 1]] = 1
        item_encode[:, self.node_nums['user']:][is_correct == 0, source_ids[is_correct == 0]] = 1
        return user_encode, item_encode
    
    def message_aggrate(self, node_type, message_type, status1, status2, node_ids, timestamps, response=None):
        if message_type == 'pref':
            message = self.pref_message[node_type](status1, status2)
            unique_node_ids, merged_message, merged_node_t = self.merged_by_segment_csr(node_ids, message, timestamps)
        else:
            message = self.knowl_message[node_type](status1, status2, response)
            unique_node_ids, merged_message, merged_node_t = self.merged_by_segment_csr(node_ids, message, timestamps)
        return unique_node_ids, merged_message, merged_node_t

    def iterative_update(self, node_type, message_type, message, hidden, cell, delta_t, hidden2=None):
        if message_type == 'pref':
            updated_hidden, updated_cell = self.pref_updater[node_type](message, hidden, cell, delta_t, hidden2)
        else:
            updated_hidden, updated_cell = self.knowl_updater[node_type](message, hidden, cell, delta_t, hidden2)
        return updated_hidden, updated_cell

    def forward(self, graph:HeteroData, interactions:list):
        self.out_graph = graph.clone()
        if self.batch_size==0:
            batch_interactions = [interactions]
        else:
            batch_interactions = torch.split(interactions, self.batch_size)
        for batch_data in tqdm(batch_interactions, desc='DGNN Loop', position=3, leave=False):
            pref_user_ids = batch_data[:,0].long()
            pref_item_ids = batch_data[:,1].long()
            pref_timestamps = batch_data[:,2].float().unsqueeze(1)
            if self.state_mode in ['both', 'pref']:
                user_pref_status = self.out_graph['user'].preference[pref_user_ids,:]     
                item_pref_status = self.out_graph['concept'].preference[pref_item_ids,:]

                unique_pref_user_ids, merged_user_pref_message, merged_user_curr_t = self.message_aggrate('user', 'pref', user_pref_status, item_pref_status, 
                                                                                                          pref_user_ids, pref_timestamps)
                unique_pref_item_ids, merged_item_pref_message, merged_item_curr_t = self.message_aggrate('concept', 'pref', item_pref_status, user_pref_status, 
                                                                                                          pref_item_ids, pref_timestamps)

                pref_user_delta_t = merged_user_curr_t - self.node_timestamp['pref']['user'][unique_pref_user_ids]
                trans_pref_user_delta_t = self.decayer(pref_user_delta_t)
                pref_item_delta_t = merged_item_curr_t - self.node_timestamp['pref']['concept'][unique_pref_item_ids]
                trans_pref_item_delta_t = self.decayer(pref_item_delta_t)

                user_pref_cell = self.out_graph['user'].pref_cell[unique_pref_user_ids,:]
                user_pref_hidden = self.out_graph['user'].pref_hidden[unique_pref_user_ids,:]
                if self.cross: user_knowl_hidden2 = self.out_graph['user'].knowl_hidden[unique_pref_user_ids,:]
                else: user_knowl_hidden2 = None
                
                item_pref_cell = self.out_graph['concept'].pref_cell[unique_pref_item_ids,:]
                item_pref_hidden = self.out_graph['concept'].pref_hidden[unique_pref_item_ids,:]
                if self.cross: item_knowl_hidden2 = self.out_graph['concept'].knowl_hidden[unique_pref_item_ids,:]
                else: item_knowl_hidden2 = None

                updated_user_pref_hidden, updated_user_pref_cell = self.iterative_update('user', 'pref', merged_user_pref_message, 
                                                                                         user_pref_hidden, user_pref_cell, 
                                                                                         trans_pref_user_delta_t, user_knowl_hidden2)
                updated_item_pref_hidden, updated_item_pref_cell = self.iterative_update('concept', 'pref', merged_item_pref_message, 
                                                                                         item_pref_hidden, item_pref_cell,
                                                                                         trans_pref_item_delta_t, item_knowl_hidden2)
                
                updated_user_pref_status = self.pref_combiner(updated_user_pref_hidden, user_pref_hidden)
                updated_item_pref_status = self.pref_combiner(updated_item_pref_hidden, item_pref_hidden)

            uq_index = batch_data[:,3]==2
            self.has_up = True
            if uq_index.sum()==0:
                self.has_up = False
            knowl_user_ids = batch_data[uq_index,0].long()
            knowl_item_ids = batch_data[uq_index,1].long()
            knowl_timestamps = batch_data[uq_index,2].float().unsqueeze(1)
            is_correct = batch_data[uq_index,4].long()
            if self.state_mode in ['both', 'knowl'] and self.has_up:
                user_knowl_status = self.out_graph['user'].knowledge[knowl_user_ids,:]
                item_knowl_status = self.out_graph['concept'].knowledge[knowl_item_ids,:]

                user_knowl_encode, item_knowl_encode = self.knowl_encode(knowl_user_ids, knowl_item_ids, is_correct)
                unique_knowl_user_ids, merged_user_knowl_message, merged_user_curr_t = self.message_aggrate('user', 'knowl', user_knowl_status, item_knowl_status, 
                                                                                                            knowl_user_ids, knowl_timestamps, response=user_knowl_encode)
                unique_knowl_item_ids, merged_item_knowl_message, merged_item_curr_t = self.message_aggrate('concept', 'knowl', item_knowl_status, user_knowl_status, 
                                                                                                            knowl_item_ids, knowl_timestamps, response=item_knowl_encode)
                        
                knowl_user_delta_t = merged_user_curr_t - self.node_timestamp['knowl']['user'][unique_knowl_user_ids]
                transed_source_delta_t = self.decayer(knowl_user_delta_t)
                knowl_item_delta_t = merged_item_curr_t - self.node_timestamp['knowl']['concept'][unique_knowl_item_ids]
                transed_target_delta_t = self.decayer(knowl_item_delta_t)

                user_knowl_cell = self.out_graph['user'].knowl_cell[unique_knowl_user_ids,:]
                user_knowl_hidden = self.out_graph['user'].knowl_hidden[unique_knowl_user_ids,:]
                if self.cross: user_pref_hidden2 = self.out_graph['user'].pref_hidden[unique_knowl_user_ids,:]
                else: user_pref_hidden2 = None

                item_knowl_cell = self.out_graph['concept'].knowl_cell[unique_knowl_item_ids,:]
                item_knowl_hidden = self.out_graph['concept'].knowl_hidden[unique_knowl_item_ids,:]
                if self.cross: item_pref_hidden2 = self.out_graph['concept'].pref_hidden[unique_knowl_item_ids,:]
                else: item_pref_hidden2 = None

                updated_user_knowl_hidden, updated_user_knowl_cell = self.iterative_update('user', 'knowl', merged_user_knowl_message, 
                                                                                            user_knowl_hidden, user_knowl_cell, 
                                                                                            transed_source_delta_t, user_pref_hidden2)
                updated_item_knowl_hidden, updated_item_knowl_cell = self.iterative_update('concept', 'knowl', merged_item_knowl_message, 
                                                                                            item_knowl_hidden, item_knowl_cell,
                                                                                            transed_target_delta_t, item_pref_hidden2)

                updated_user_knowl_status = self.knowl_combiner(updated_user_knowl_hidden, user_knowl_hidden)
                updated_item_knowl_status = self.knowl_combiner(updated_item_knowl_hidden, item_knowl_hidden)

            if self.state_mode in ['both', 'pref']:
                self.out_graph['user'].preference[unique_pref_user_ids,:] = updated_user_pref_status
                self.out_graph['user'].pref_hidden[unique_pref_user_ids,:] = updated_user_pref_hidden
                self.out_graph['user'].pref_cell[unique_pref_user_ids,:] = updated_user_pref_cell
                self.out_graph['concept'].preference[unique_pref_item_ids,:] = updated_item_pref_status
                self.out_graph['concept'].pref_hidden[unique_pref_item_ids,:] = updated_item_pref_hidden
                self.out_graph['concept'].pref_cell[unique_pref_item_ids,:] = updated_item_pref_cell

                unique_batch_pref_user, pref_user_inverse_indices = torch.unique(pref_user_ids, return_inverse=True)
                unique_batch_pref_item, pref_item_inverse_indices = torch.unique(pref_item_ids, return_inverse=True)
                pref_user_max_curr_t = scatter_max(pref_timestamps, pref_user_inverse_indices, dim=0)[0]
                pref_item_max_curr_t = scatter_max(pref_timestamps, pref_item_inverse_indices, dim=0)[0]
                with torch.no_grad():
                    self.node_timestamp['pref']['user'][unique_batch_pref_user] = pref_user_max_curr_t
                    self.node_timestamp['pref']['concept'][unique_batch_pref_item] = pref_item_max_curr_t

            if self.state_mode in ['both', 'knowl'] and self.has_up:
                self.out_graph['user'].knowledge[unique_knowl_user_ids,:] = updated_user_knowl_status
                self.out_graph['user'].knowl_hidden[unique_knowl_user_ids,:] = updated_user_knowl_hidden
                self.out_graph['user'].knowl_cell[unique_knowl_user_ids,:] = updated_user_knowl_cell
                self.out_graph['concept'].knowledge[unique_knowl_item_ids,:] = updated_item_knowl_status
                self.out_graph['concept'].knowl_hidden[unique_knowl_item_ids,:] = updated_item_knowl_hidden
                self.out_graph['concept'].knowl_cell[unique_knowl_item_ids,:] = updated_item_knowl_cell

                unique_batch_knowl_user, knowl_user_inverse_indices = torch.unique(knowl_user_ids, return_inverse=True)
                unique_batch_knowl_item, knowl_item_inverse_indices = torch.unique(knowl_item_ids, return_inverse=True)
                knowl_user_max_curr_t = scatter_max(knowl_timestamps, knowl_user_inverse_indices, dim=0)[0]
                knowl_item_max_curr_t = scatter_max(knowl_timestamps, knowl_item_inverse_indices, dim=0)[0]
                with torch.no_grad():
                    self.node_timestamp['knowl']['user'][unique_batch_knowl_user] = knowl_user_max_curr_t
                    self.node_timestamp['knowl']['concept'][unique_batch_knowl_item] = knowl_item_max_curr_t

            if self.neighbors > 0:
                batch_last_t = batch_data[-1,2]
                if self.state_mode in ['both', 'pref']:
                    self.propagation(unique_pref_user_ids, 'user', 'concept', batch_last_t, merged_user_pref_message, message_type='pref')
                    self.propagation(unique_pref_item_ids, 'concept', 'user', batch_last_t, merged_item_pref_message, message_type='pref')
                if self.state_mode in ['both', 'knowl'] and self.has_up:
                    self.propagation(unique_knowl_user_ids, 'user', 'concept', batch_last_t, merged_user_knowl_message, message_type='knowl')
                    self.propagation(unique_knowl_item_ids, 'concept', 'user', batch_last_t, merged_item_knowl_message, message_type='knowl')
            
            if self.neighbors > 0:
                if self.state_mode in ['both', 'pref']:
                    unique_batch_edge, edge_inverse_indices = torch.unique(batch_data[:,:2].long(), return_inverse=True, dim=0)
                    merged_edge_current_t = scatter_max(batch_data[:,2].float(), edge_inverse_indices, dim=0)[0]
                    self.neighbor_loader.insert(unique_batch_edge, merged_edge_current_t)
                if self.state_mode in ['knowl'] and self.has_up:
                    unique_batch_edge, edge_inverse_indices = torch.unique(batch_data[uq_index,:2].long(), return_inverse=True, dim=0)
                    merged_edge_current_t = scatter_max(batch_data[uq_index,2].float(), edge_inverse_indices, dim=0)[0]
                    self.neighbor_loader.insert(unique_batch_edge, merged_edge_current_t)
        
        return self.out_graph

    def get_neighbors(self, node_ids, node_type, message):
        neighbors = self.neighbor_loader(node_ids, node_type)
        mask = (neighbors!=-1).all(dim=1)
        valid_node_ids = node_ids[mask]
        if valid_node_ids.shape[0]==0:
            return None, None, None
        
        valid_neighbor_ids = neighbors[mask]
        valid_message = message[mask]
        return valid_node_ids, valid_neighbor_ids, valid_message
        
    def get_att_pref_score(self, node_id, node_type, neighbors, neighbor_type):
        neighbors_general = self.out_graph[neighbor_type].preference[neighbors,:]
        node_general = self.out_graph[node_type].preference[node_id,:]
        node_reps = node_general.unsqueeze(1).repeat(1, neighbors_general.size(1), 1)
        return self.pref_nei_att[node_type](node_reps, neighbors_general)
    
    def get_att_knowl_score(self, node_id, node_type, neighbors, neighbor_type):
        neighbors_general = self.out_graph[neighbor_type].knowledge[neighbors,:]
        node_general = self.out_graph[node_type].knowledge[node_id,:]
        node_reps = node_general.unsqueeze(1).repeat(1, neighbors_general.size(1), 1)
        return self.knowl_nei_att[node_type](node_reps, neighbors_general)
    
    def propagation(self, node_ids, node_type, neighbor_type, current_t, message=None, message_type='pref') -> None:

        valid_node_ids, valid_neighbor_ids, valid_message = self.get_neighbors(node_ids, node_type, message)
        if valid_neighbor_ids is None or valid_neighbor_ids.shape[1]==0:
            return None

        valid_message = valid_message.unsqueeze(1).repeat(1, valid_neighbor_ids.size(1), 1)
        if message_type=='pref':
            att_score = self.get_att_pref_score(valid_node_ids, node_type, valid_neighbor_ids, neighbor_type)
        else:
            att_score = self.get_att_knowl_score(valid_node_ids, node_type, valid_neighbor_ids, neighbor_type)
        att_score = att_score.view(valid_message.size(0), valid_neighbor_ids.size(1), 1)
        sum_message = valid_message * att_score

        flat_neighbor_ids = valid_neighbor_ids.flatten()
        flat_sum_message = sum_message.view((-1, self.message_dims))
        unique_neighbor_ids, merged_nei_message = self.merged_neighbors(flat_neighbor_ids, flat_sum_message)
        if torch.any(flat_neighbor_ids == -1):
            unique_neighbor_ids = unique_neighbor_ids[1:]
            merged_nei_message = merged_nei_message[1:]

        if message_type=='pref':
            unique_nei_pref_hidden = self.out_graph[neighbor_type].pref_hidden[unique_neighbor_ids,:]
            unique_nei_pref_cell = self.out_graph[neighbor_type].pref_cell[unique_neighbor_ids,:]
        else:
            unique_nei_knowl_hidden = self.out_graph[neighbor_type].knowl_hidden[unique_neighbor_ids,:]
            unique_nei_knowl_cell = self.out_graph[neighbor_type].knowl_cell[unique_neighbor_ids,:]       

        if message_type=='pref':
            if self.args.trans_message:
                merged_nei_message = self.trans_pref_message(merged_nei_message)
            updated_nei_pref_cell = unique_nei_pref_cell + merged_nei_message
            updated_nei_pref_hidden = self.act(updated_nei_pref_cell)
            updated_nei_pref_status = self.pref_combiner(updated_nei_pref_hidden, unique_nei_pref_hidden)
        else:
            if self.args.trans_message:
                merged_nei_message = self.trans_knowl_message(merged_nei_message)
            updated_nei_knowl_cell = unique_nei_knowl_cell + merged_nei_message
            updated_nei_knowl_hidden = self.act(updated_nei_knowl_cell)
            updated_nei_knowl_status = self.knowl_combiner(updated_nei_knowl_hidden, unique_nei_knowl_hidden)

        if message_type=='pref':
            self.out_graph[neighbor_type].preference[unique_neighbor_ids,:] = updated_nei_pref_status
            self.out_graph[neighbor_type].pref_hidden[unique_neighbor_ids,:] = updated_nei_pref_hidden
            self.out_graph[neighbor_type].pref_cell[unique_neighbor_ids,:] = updated_nei_pref_cell
        else:
            self.out_graph[neighbor_type].knowledge[unique_neighbor_ids,:] = updated_nei_knowl_status
            self.out_graph[neighbor_type].knowl_hidden[unique_neighbor_ids,:] = updated_nei_knowl_hidden
            self.out_graph[neighbor_type].knowl_cell[unique_neighbor_ids,:] = updated_nei_knowl_cell

        with torch.no_grad():
            self.node_timestamp[message_type][neighbor_type][unique_neighbor_ids] = current_t

    def merged_neighbors(self, neighbor_ids, message):
        sorted_neighbor_ids, sorted_neighbor_indices = torch.sort(neighbor_ids)
        sorted_message = message[sorted_neighbor_indices]

        unique_neighbor_ids, unique_counts = torch.unique_consecutive(sorted_neighbor_ids, return_counts=True) 
        node_segment_indices = torch.cumsum(unique_counts, dim=0) - unique_counts 
        node_segment_indices = torch.cat((node_segment_indices, torch.tensor([len(sorted_neighbor_ids)], device=self.device))) 
        
        merged_message = segment_csr(sorted_message, node_segment_indices, reduce=self.msg_agg)
        return unique_neighbor_ids, merged_message