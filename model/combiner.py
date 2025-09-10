import torch.nn as nn

class Combiner(nn.Module):
	def __init__(self, hidden_dims, node_dims, act=None):
		super(Combiner, self).__init__()
		self.h2o = nn.Linear(hidden_dims, node_dims)
		self.l2o = nn.Linear(hidden_dims, node_dims)
		self.act = act

		self._init_param()

	def _init_param(self):
		nn.init.xavier_uniform_(self.h2o.weight)
		nn.init.zeros_(self.h2o.bias)
		nn.init.xavier_uniform_(self.l2o.weight)
		nn.init.zeros_(self.l2o.bias)

	def forward(self, cur_ht, prev_ht):
		node_output = self.h2o(cur_ht) + self.l2o(prev_ht)
		node_output = self.act(node_output)
		return node_output
