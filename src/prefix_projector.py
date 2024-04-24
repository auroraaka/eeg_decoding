import torch.nn as nn

class PrefixProjector(nn.Module):
    def __init__(self, in_features, out_features, pool_size):
        super(PrefixProjector, self).__init__()
        self.linear_proj = nn.Linear(in_features, out_features)
        self.avg_pool = nn.AvgPool1d(kernel_size=pool_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.linear_proj(x)
        x = x.permute(0, 2, 1)
        x = self.avg_pool(x)
        x = x.permute(0, 2, 1)
        return x

def build_prefix_projector(config):
    in_features = config.encoder.in_features
    out_features = config.encoder.out_features
    pool_size = config.encoder.pool_size
    projector = PrefixProjector(in_features, out_features, pool_size)
    return projector

