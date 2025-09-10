import torch.nn as nn

class EdgeMessage(nn.Module):
	def __init__(self, node_dims, message_dims, relation_dims=None, act=None):
		super(EdgeMessage,self).__init__()
		self.s2m = nn.Linear(node_dims, message_dims)
		self.t2m = nn.Linear(node_dims, message_dims)
		if relation_dims is not None:
			self.r2m = nn.Linear(relation_dims, message_dims)

		self.act = act

		self._init_param()

	def _init_param(self):
		nn.init.xavier_uniform_(self.s2m.weight)
		nn.init.zeros_(self.s2m.bias)
		nn.init.xavier_uniform_(self.t2m.weight)
		nn.init.zeros_(self.t2m.bias)
		if hasattr(self, 'r2m'):
			nn.init.xavier_uniform_(self.r2m.weight)
			nn.init.zeros_(self.r2m.bias)

	def forward(self, source_node, target_node, relation=None):
		if relation is None:
			message = self.s2m(source_node) + self.t2m(target_node)
		else:
			message = self.s2m(source_node) + self.t2m(target_node) + self.r2m(relation)
		message = self.act(message)
		return message
  