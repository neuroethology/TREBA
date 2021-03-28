import torch
import torch.nn.functional as F

from .core import Distribution


class Multinomial(Distribution):

    def __init__(self, log_probs):
        super().__init__()

        self.log_probs = log_probs

    def sample(self):
        inds = torch.multinomial(torch.exp(self.log_probs), 1)
        samples = torch.zeros(self.log_probs.size())
        samples.scatter_(-1, inds, 1)
        return samples
        
    def log_prob(self, value):
        assert value.size() == self.log_probs.size()
        return torch.sum(self.log_probs*value)
