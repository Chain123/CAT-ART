"""
Overall model for multi-domain user modeling and embedding generating.
"""
import torch.nn as nn
import bert
import torch
import numpy as np
from .DomainRepresentation import MulDomainEmbed
from .generator import Generator
# from .UserTendency import UserTendency


class GenderModel(nn.Module):

    def __init__(self, Mid, out):
        super(GenderModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(Mid[0], Mid[1]),
            nn.BatchNorm1d(Mid[1]),
            nn.PReLU(),
            nn.Linear(Mid[1], Mid[2]),
            nn.BatchNorm1d(Mid[2]),
            nn.PReLU(),
            nn.Linear(Mid[2], out)  # output 1 for gender prediction.
        )

    def forward(self, x):
        """
        x: user embeddings in multiple domains
        """
        return self.layers(x)


class ummd(nn.Module):
    """
    Overall module
    """

    def __init__(self, hyper_param, embed_class, device, pre_path=None, threshold=0.5,
                 fix_pos=False):
        super(ummd, self).__init__()

        # Single domain embedding modules.
        self.domain_embed = MulDomainEmbed(embed_class, 5, hyper_param.n_item,
                                           hyper_param.latent_dim, embed_class,
                                           hyper_param.length, device, path=pre_path)
        # Generator Module
        self.Generator = Generator(hyper_param.latent_dim, int(hyper_param.latent_dim / 64),
                                   0.2, hyper_param.n_layer, fix_pos=fix_pos)
        self.Domain_padding = nn.Embedding(5, hyper_param.latent_dims[-1])
        nn.init.normal_(self.Domain_padding.weight, std=0.01)

        # User Tendency module
        self.TendencyModule = UserTendency(threshold=threshold)

        # Output Module
        self.OutLayer = GenderModel(hyper_param.MidDims, 1)

        # Attributes
        self.device = device
        self.arg = hyper_param

    def forward_domain(self, tag, mask, domain):
        """ user embedding for the domain-th domain """
        return self.domain_embed.DomainEmbed(tag[:, domain, :], mask[:, domain], domain)

    def forward(self, x, mask_tag, mask_domain):
        """ Main forward path
        x: user behavior (tag set) at each domains
        mask_tag: tag-level mask for each domain
        mask_domain: domain-level mask indicating missing domains for each user
        """
        x_embed = self.domain_embed(x, mask_tag)
        g_embed = self.Generator(x_embed, mask_domain)
        m_embed = self.TendencyModule(x_embed, g_embed,
                                      self.Domain_padding(torch.from_numpy(np.arange(5))))
        logits = self.OutLayer(m_embed)
        return logits
