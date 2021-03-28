import math
import torch

from .core import Distribution


class Normal(Distribution):

    logvar_min = -16
    logvar_max = 16

    def __init__(self, mean, logvar):
        super().__init__()

        self.mean = mean
        self.logvar = torch.clamp(logvar, self.logvar_min, self.logvar_max)

    def sample(self, temperature=1.0):
        std = torch.exp(self.logvar / 2)
        eps = torch.randn_like(std)
        return self.mean + eps * std * temperature

    def log_prob(self, value):
        pi = torch.FloatTensor([math.pi]).to(value.device)
        nll_element = (value - self.mean).pow(2) / \
            torch.exp(self.logvar) + self.logvar + torch.log(2 * pi)
        return -0.5 * torch.sum(nll_element)

    @staticmethod
    def kl_divergence(normal_1, normal_2=None, free_bits=0.0):
        '''
        Computes the kl-divergence between two normal distributions.

        Args:
            normal_1 (Normal): first normal distribution
            normal_2 (Normal): second normal distribution (assume standard normal if not provided)
        '''

        assert isinstance(normal_1, Normal)
        mean_1, logvar_1 = normal_1.mean, normal_1.logvar

        if normal_2 is not None:
            assert isinstance(normal_2, Normal)
            mean_2, logvar_2 = normal_2.mean, normal_2.logvar

            kld_elements = 0.5 * (logvar_2 - logvar_1 +
                                  (torch.exp(logvar_1) + (mean_1 - mean_2).pow(2)) /
                                  torch.exp(logvar_2) - 1)
        else:
            kld_elements = -0.5 * \
                (1 + logvar_1 - mean_1.pow(2) - torch.exp(logvar_1))

        # Prevent posterior collapse with free bits
        if free_bits > 0.0:
            _lambda = free_bits * \
                torch.ones(kld_elements.size()).to(kld_elements.device)
            kld_elements = torch.max(kld_elements, _lambda)

        kld = torch.sum(kld_elements, dim=-1)

        return kld
