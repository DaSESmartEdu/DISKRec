import torch
from torch import nn

class RateLayer(nn.Module):
    def __init__(self, user_num, item_num, latent_dim, output_dim):
        super(RateLayer, self).__init__()
        self.name = 'RateLayer'
        self.vars = nn.ParameterDict()
        self.user_latent = nn.Parameter(torch.randn(user_num, latent_dim))
        self.item_latent = nn.Parameter(torch.randn(item_num, latent_dim))
        self.user_specific = nn.Parameter(torch.randn(item_num, output_dim))
        self.item_specific = nn.Parameter(torch.randn(user_num, output_dim))
        self.user_bias = nn.Parameter(torch.zeros(user_num, 1))
        self.item_bias = nn.Parameter(torch.zeros(item_num, 1))
        self.alpha1 = nn.Parameter(torch.tensor(1.0))
        self.alpha2 = nn.Parameter(torch.tensor(1.0))

        self._init_xavier()

    def _init_xavier(self):
        nn.init.xavier_uniform_(self.user_latent)
        nn.init.xavier_uniform_(self.item_latent)
        nn.init.xavier_uniform_(self.user_specific)
        nn.init.xavier_uniform_(self.item_specific)
        nn.init.zeros_(self.user_bias)
        nn.init.zeros_(self.item_bias)

    def forward(self, user, item):
        rate_matrix1 = torch.matmul(self.user_latent, self.item_latent.t())
        
        u_matrix = self.alpha1 * (torch.matmul(user, self.user_specific.t()) + self.user_bias)
        i_matrix = self.alpha2 * (torch.matmul(item, self.item_specific.t()) + self.item_bias).t()

        rate_matrix2 = rate_matrix1 + u_matrix + i_matrix
        return rate_matrix2

