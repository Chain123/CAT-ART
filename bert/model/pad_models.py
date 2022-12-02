import torch.nn as nn
from bert.model.transformer import TransformerBlock
import bert
import torch
import numpy as np


class avg_pad(nn.Module):

    def __init__(self, n_domain, latent_dim, device):
        super(avg_pad).__init__()
        self.n_domain = n_domain
        self.pad_embedding = nn.Embedding(self.n_domain, latent_dim)
        nn.init.normal_(self.pad_embedding.weight, std=0.01)
        self.device = device

    def forward(self, domain_embed, domain, domain_mask):
        if domain_mask is not None:
            domain_tensor = torch.from_numpy(np.array([domain])).to(self.device)
            domain_mask = domain_mask[:, domain]
            domain_mask_index = (domain_mask == 1).nonzero(as_tuple=True)[0]
            if len(domain_mask_index) > 0:
                domain_embed[domain_mask_index] = self.pad_embedding(domain_tensor)

        return domain_embed


class transformer_pad(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.position = bert.model.embedding.PositionalEmbedding(d_model=args.latent_dims[0])
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(args.latent_dims[0], args.n_head,
                              args.latent_dims[0] * 4, args.attn_dropout) for _ in range(args.n_layer)])

    def forward(self, x, mask):
        all_embedding = x + self.position(5)
        mask_attn = mask.unsqueeze(1).repeat(1, mask.size(1), 1).unsqueeze(1)
        for transformer in self.transformer_blocks:
            all_embedding = transformer.forward(all_embedding, mask_attn)

        return all_embedding



