import torch
import torch.nn as nn

class LinearAttention(nn.Module):
    def __init__(self, message_dims, hidden_dims):
        super(LinearAttention,self).__init__()
        self.hidden_dims = hidden_dims
        self.mlp = nn.Linear(message_dims+hidden_dims, 1)

        self.init_param()

    def init_param(self):
        nn.init.xavier_uniform_(self.mlp.weight)
        nn.init.zeros_(self.mlp.bias)

    def forward(self, message, a_hidden, b_hidden):
        message_copy = message.unsqueeze(1).expand(message.size(0), 2, message.size(1))
        previous_hidden = torch.cat([a_hidden.unsqueeze(dim=1), b_hidden.unsqueeze(dim=1)], dim=1)
        att_coef_hidden = torch.softmax(self.mlp(torch.cat([message_copy, previous_hidden], dim=2)).squeeze(-1), dim=1) 
        att_coef_hidden_ = att_coef_hidden.unsqueeze(dim=2).expand(att_coef_hidden.size(0), att_coef_hidden.size(1), self.hidden_dims)
        out = torch.sum(att_coef_hidden_ * previous_hidden, dim=1)
        return out

class node_LSTM(nn.Module):
	def __init__(self, message_dims, hidden_dims, decay=True, cross=True):
		super(node_LSTM, self).__init__()
		self.decay = decay
		self.cross = cross
		
		if cross: self.cross_fusion = LinearAttention(message_dims, hidden_dims)

		self.m2h = nn.Linear(message_dims, 4*hidden_dims)
		self.h2h = nn.Linear(hidden_dims, 4*hidden_dims)
		self.c2s = nn.Sequential(nn.Linear(hidden_dims, hidden_dims), nn.Tanh())
		self.sigmoid = nn.Sigmoid()
		self.tanh = nn.Tanh()

		self._init_param()

	def _init_param(self):
		nn.init.xavier_uniform_(self.m2h.weight)
		nn.init.zeros_(self.m2h.bias)
		nn.init.xavier_uniform_(self.h2h.weight)
		nn.init.zeros_(self.h2h.bias)
		nn.init.xavier_uniform_(self.c2s[0].weight)
		nn.init.zeros_(self.c2s[0].bias)

	def forward(self, message, hidden, cell, transed_delta_t=None, hidden2=None):
		if self.decay and transed_delta_t is not None:
			cell_short = self.c2s(cell)
			cell_new = cell - cell_short + cell_short * transed_delta_t
		else:
			cell_new = cell
			 
		if self.cross and hidden2 is not None:
			hidden_new = self.cross_fusion(message, hidden, hidden2)
		else:
			hidden_new = hidden
		
		gates = self.m2h(message) + self.h2h(hidden_new)
		ingate, forgate, cellgate, outgate = gates.chunk(4, -1)
		
		ingate = self.sigmoid(ingate)
		forgate = self.sigmoid(forgate)
		cellgate = self.tanh(cellgate)
		outgate = self.sigmoid(outgate)

		cell_out = forgate * cell_new + ingate * cellgate
		hidden_out = outgate * self.tanh(cell_out)

		return hidden_out, cell_out


	