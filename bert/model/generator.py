import torch.nn as nn

from .transformer import TransformerBlock
from .embedding import PositionalEmbedding


class Generator(nn.Module):

    def __init__(self, hidden, attn_heads, dropout, n_layers, fix_pos):
        super(Generator, self).__init__()
        if fix_pos:
            self.position = PositionalEmbedding(d_model=hidden)
        else:
            self.position = nn.Embedding(5, hidden)
            nn.init.normal_(self.position.weight, std=0.01)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, mask=None):
        """
        x:    user embeddings at each domain, [B, 5, latent_dim]
        mask: indicators: [B, 5], 1-indicating an missing domain, 0-mean an existing domain.
        """
        # pre-process input embedding according to mask
        # positional embedding.

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, None)
        return x
