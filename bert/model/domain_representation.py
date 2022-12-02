import torch.nn as nn
import torch
from .vq import Quantize
import numpy as np
import bert
from torch import LongTensor as LT
from torch import FloatTensor as FT


class domain_model(nn.Module):
    """
    Get user's representation in a specific domain.
    available data: users long term interaction with "tags"
    MF to get user's embedding:
        user lookup table: when the number of user is not too big.
        item-based: only item embeddings. and a function from item embed to user embedding
        mixed: short user embedding.
    """

    def __init__(self, n_user=None, n_item=None, codebook_size=256, latent_dim=50, user_embed="user",
                 pad_ind=0, seq_len=None):
        """
        """
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.user_embed_method = user_embed
        self.pad_idx = pad_ind
        if self.codebook_size > 0:
            self.quantize = Quantize(latent_dim, codebook_size)

        self.item_embed = nn.Embedding(n_item + 1, latent_dim)
        nn.init.normal_(self.item_embed.weight, std=0.01)
        # TODO more tag based user embedding methods.
        if user_embed == "user":
            self.user_embed = nn.Embedding(n_user, latent_dim)
            nn.init.normal_(self.user_embed.weight, std=0.01)
        elif user_embed == "item":
            self.user_embed = None
            # TODO what is the best activation function?
            self.map = nn.Sequential(nn.Linear(latent_dim, latent_dim * 2),
                                     nn.ReLU(),
                                     nn.Linear(latent_dim * 2, latent_dim))
        elif user_embed == "half":
            self.user_embed = nn.Embedding(n_user, int(latent_dim * 0.2))
            self.map = nn.Sequential(nn.Linear(latent_dim, latent_dim * 2),
                                     nn.ReLU(),
                                     nn.Linear(latent_dim * 2, int(latent_dim * 0.8)))
        elif user_embed == "maxpool":
            self.MaxPoolLayer = nn.MaxPool1d(seq_len, stride=1)
        elif user_embed == "attn":
            self.MaxPoolLayer = nn.MaxPool1d(seq_len, stride=1)
            # attention layer
            self.attention = bert.model.Attention()
            self.norm = bert.model.utils.LayerNorm(latent_dim)
            self.dropout = nn.Dropout(0.2)
        else:
            print("ERROR! NO USRE EMBEDDING METHOD SPECIFIED!")

    def get_user_embed(self, user_id, mean_item_embed):
        """
        :param user_id:             [B, 1]
        :param mean_item_embed:     [B, latent_dim]
        :return:                    [B, latent_dim]
        self.user_embed_method:
            item: mean item embeddings
            maxpool: max pooling of the
        """
        if self.user_embed_method == "user":
            return self.user_embed(user_id)
        elif self.user_embed_method == "item":
            return self.map(mean_item_embed)
        else:
            return torch.cat((self.user_embed(user_id), self.map(mean_item_embed)), 1)

    def MaxPool(self, tags):
        return self.MaxPoolLayer(tags.permute(0, 2, 1)).permute(0, 2, 1)

    def domain_embedding(self, user_id, interacted_items, mask=None):
        """
        :param user_id:
        :param interacted_items:
        :param mask:
        :return:
        """
        # get user embedding
        if mask is None:
            mask = interacted_items == self.pad_idx  # [B, max_len]

        if self.user_embed_method == "user":
            user_embed = self.user_embed(user_id)
        elif self.user_embed_method == "item":
            mask = (1 - mask.to(int)).unsqueeze(2)  # [B, K, 1]
            '''mask out padding and then point-wise mean'''
            mean_item_embed = self.item_embed(interacted_items)  # [B, k, latent_dim]
            mean_item_embed = torch.sum(mean_item_embed * mask, 1) / torch.sum(mask, 1)
            user_embed = self.map(mean_item_embed)
        elif self.user_embed_method == 'half':
            ''' a smaller size trainable user embedding matrix with the tag embedding'''
            mask = (1 - mask.to(int)).unsqueeze(2)  # [B, K, 1]
            mean_item_embed = self.item_embed(interacted_items)  # [B, k, latent_dim]
            mean_item_embed = torch.sum(mean_item_embed * mask, 1) / torch.sum(mask, 1)
            user_embed = torch.cat((self.user_embed(user_id), self.map(mean_item_embed)), 1)
        elif self.user_embed_method == 'maxpool':  #
            ''' MalPool on non-padding tag embeddings'''
            masked_embedding = self.item_embed(interacted_items).masked_fill(mask.unsqueeze(2),
                                                                             value=torch.tensor(-1e9))
            # [B, seq, latent_dim]
            # print(masked_embedding.size())
            user_embed = self.MaxPool(masked_embedding)
        elif self.user_embed_method == 'attn':
            ''' attention on tag list: then MaxPool'''
            mask_attn = (mask > 0).unsqueeze(1).repeat(1, mask.size(1), 1).unsqueeze(1)  # copy from bert.py

            embedding = self.item_embed(interacted_items)   # [B, max_len, latent_dim]
            attn_embed, _ = self.attention.forward(embedding, embedding, embedding, mask=mask_attn,
                                                   dropout=self.dropout)
            # query, key, value, mask=None, dropout=None
            masked_embedding = (embedding + attn_embed).masked_fill(mask.unsqueeze(2),
                                                                    value=torch.tensor(-1e9))
            user_embed = self.MaxPool(masked_embedding)
        else:
            user_embed = None
            print("Error, no valid user embedding method specified!")

        # vector quantification
        if user_embed is not None and self.codebook_size > 0:
            # TODO disable VQ book from updating itself in UMMD process. need to check in train and ummd cases.
            if not self.training:
                self.quantize.eval()
            quant_user, diff, id_t = self.quantize(user_embed)
        else:    # None and zero tensors.
            quant_user = user_embed
            diff = torch.from_numpy(np.zeros(quant_user.shape[0]))
            id_t = torch.from_numpy(np.zeros(quant_user.shape[0]))

        return quant_user, diff, id_t, user_embed

    def forward(self, user_id, interacted_items, pos, neg, mask=None):
        """
        :param user_id:             [B, 1]
        :param interacted_items:    [B, K] (randomly sample K) padding.
        :param pos:                 []
        :param neg:
        :param mask:
        :return:
        """
        mask = None    # use the mask we calculate in the self.domain_embedding method.
        quant_user, diff, id_t, user_embed = self.domain_embedding(user_id, interacted_items, mask=mask)
        pos_item = self.item_embed(pos)
        neg_item = self.item_embed(neg)

        return quant_user, pos_item, neg_item, diff, user_embed


class Bundler(nn.Module):

    def forward(self, data):
        raise NotImplementedError

    def forward_i(self, data):
        raise NotImplementedError

    def forward_o(self, data):
        raise NotImplementedError


class Word2Vec(Bundler):

    def __init__(self, vocab_size=20000, embedding_size=300, padding_idx=0):
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.ivectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ovectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ivectors.weight = nn.Parameter(torch.cat([torch.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        self.ovectors.weight = nn.Parameter(torch.cat([torch.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True

    def forward(self, data):
        return self.forward_i(data)

    def forward_i(self, data):
        v = LT(data)
        v = v.cuda() if self.ivectors.weight.is_cuda else v
        return self.ivectors(v)

    def forward_o(self, data):
        v = LT(data)
        v = v.cuda() if self.ovectors.weight.is_cuda else v
        return self.ovectors(v)


class SGNS(nn.Module):

    def __init__(self, embedding, vocab_size=20000, n_negs=20, weights=None):
        super(SGNS, self).__init__()
        self.embedding = embedding
        self.vocab_size = vocab_size
        self.n_negs = n_negs
        self.weights = None
        if weights is not None:
            wf = np.power(weights, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)

    def forward(self, iword, owords):
        batch_size = iword.size()[0]
        context_size = owords.size()[1]
        if self.weights is not None:
            nwords = torch.multinomial(self.weights, batch_size * context_size * self.n_negs, replacement=True).view(batch_size, -1)
        else:
            nwords = FT(batch_size, context_size * self.n_negs).uniform_(0, self.vocab_size - 1).long()
        ivectors = self.embedding.forward_i(iword).unsqueeze(2)
        ovectors = self.embedding.forward_o(owords)
        nvectors = self.embedding.forward_o(nwords).neg()
        oloss = torch.bmm(ovectors, ivectors).squeeze().sigmoid().log().mean(1)
        nloss = torch.bmm(nvectors, ivectors).squeeze().sigmoid().log().view(-1, context_size, self.n_negs).sum(2).mean(1)
        nloss = torch.bmm(nvectors, ivectors).squeeze().sigmoid().log().view(-1, context_size, self.n_negs).sum(2).mean(1)
        return -(oloss + nloss).mean()
