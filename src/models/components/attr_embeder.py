import torch

class AttrEmbedding(torch.nn.Module):
    def __init__(self, emb_dim = 640, seq_len = 77, n_attrs = 40, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pa = torch.nn.Parameter(torch.rand((1, emb_dim, n_attrs)))
        self.linear = torch.nn.Linear(n_attrs, seq_len, bias=False)
        self.silu = torch.nn.SiLU()

    def forward(self, attr):
        ''' attr: int (batch_size, n_attrs) '''
        batch_size = attr.shape[0]
        cond = torch.cat([attr[i] * self.pa for i in range(batch_size)])
        cond = self.silu(cond)
        cond = self.linear(cond).transpose(-1, -2)
        return cond
