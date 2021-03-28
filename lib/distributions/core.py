import torch.nn as nn

class Distribution(nn.Module):

    def __init__(self):
        super(Distribution, self).__init__()

    def sample(self):
        raise NotImplementedError

    def log_prob(self, value):
        raise NotImplementedError
