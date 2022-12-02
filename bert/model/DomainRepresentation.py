import os

import numpy as np
import torch.nn as nn
import torch
from torch import optim
from tqdm import tqdm

import bert
from .vq import Quantize


class TagEmbedding(nn.Module):

    def __init__(self, n_item=None, latent_dim=50, pretrain=False, freeze=False, pre_dir=None):
        super(TagEmbedding, self).__init__()
        if pretrain:
            weight = torch.load(pre_dir)
            self.item_embed = nn.Embedding.from_pretrained(weight, freeze=freeze, padding_idx=0)
        else:
            self.item_embed = nn.Embedding(n_item + 1, latent_dim, padding_idx=0)
            nn.init.normal_(self.item_embed.weight, std=0.01)

    def forward(self, x):
        """
        x: [B, seq_len]
        return: [B, seq_len, latent_dim]
        """
        return self.item_embed(x)


class SingleDomainEmbedding(nn.Module):
    """
    Tag and user embedding of a given single domain
    """

    def __init__(self, n_user=None, n_item=None, latent_dim=50, user_embed="user",
                 pad_ind=0, seq_len=None, pretrain=False, freeze=False,
                 pre_dir=None, code_size=None):
        # pretrain=False, freeze=False, pre_dir=None  use in sgns pre-train case.
        super(SingleDomainEmbedding, self).__init__()
        self.code_size = code_size
        self.n_user = n_user
        self.n_item = n_item
        self.latent_dim = latent_dim
        self.user_embed_method = user_embed
        self.pad_idx = pad_ind
        self.MaxLength = seq_len

        # tag embedding table
        self.item_embed = TagEmbedding(n_item=n_item, latent_dim=latent_dim,
                                       pretrain=pretrain, freeze=freeze, pre_dir=pre_dir)
        ############################
        # tag-based user embedding #
        ############################
        if user_embed == "user":
            # user embedding table
            assert n_user is not None
            self.user_embed = TagEmbedding(n_item=n_user, latent_dim=latent_dim)
        elif user_embed == "item":
            # fully connected layer(s) on an averaged (sum) tag embedding(s).
            self.map = nn.Sequential(nn.Linear(latent_dim, latent_dim * 2),
                                     nn.ReLU(),
                                     nn.Linear(latent_dim * 2, latent_dim))
        elif user_embed == "maxpool":
            # maxpool the tag embedding
            self.MaxPoolLayer = nn.MaxPool1d(seq_len, stride=1)
        elif user_embed == "attn":
            # maxpool after several attention layers
            self.attention = bert.model.Attention()
            self.norm = bert.model.utils.LayerNorm(latent_dim)
            self.dropout = nn.Dropout(0.2)
            # Maxpool layer
            self.MaxPoolLayer = nn.MaxPool1d(seq_len, stride=1)
        elif user_embed == "attn2":
            # maxpool after several attention layers
            self.attention = bert.model.Attention()
            self.norm = bert.model.utils.LayerNorm(latent_dim)
            self.dropout = nn.Dropout(0.2)
            # Maxpool layer
            self.map = nn.Sequential(nn.Linear(latent_dim, latent_dim * 2),
                                     nn.ReLU(),
                                     nn.Linear(latent_dim * 2, latent_dim))
        else:
            print("ERROR! NO USER EMBEDDING METHOD SPECIFIED!")

    def MaxPool(self, tags):
        return self.MaxPoolLayer(tags.permute(0, 2, 1)).permute(0, 2, 1)

    def domain_embedding(self, user_id, interacted_items, mask=None):
        """get user embedding based on their interacted items"""
        if mask is None:
            mask = interacted_items == self.pad_idx  # [B, max_len]

        if self.user_embed_method == "user":
            user_embed = self.user_embed(user_id)
        elif self.user_embed_method == "item":
            mask = (1 - mask.to(int)).unsqueeze(2)  # [B, K, 1]
            '''mask out padding and then point-wise mean'''
            mean_item_embed = self.item_embed(interacted_items)  # [B, k, latent_dim]
            mean_item_embed = torch.sum(mean_item_embed * mask, 1)
            # / torch.sum(mask, 1)
            user_embed = self.map(mean_item_embed)
        elif self.user_embed_method == 'maxpool':  #
            ''' MalPool on non-padding tag embeddings'''
            masked_embedding = self.item_embed(interacted_items).masked_fill(mask.unsqueeze(2),
                                                                             value=torch.tensor(-1e9))
            # [B, seq, latent_dim]
            user_embed = torch.squeeze(self.MaxPool(masked_embedding))
        elif self.user_embed_method == 'attn':
            ''' attention on tag list: then MaxPool'''
            mask_attn = (mask > 0).unsqueeze(1).repeat(1, mask.size(1), 1)

            embedding = self.item_embed(interacted_items)  # [B, max_len, latent_dim]
            attn_embed, _ = self.attention.forward(embedding, embedding, embedding, mask=mask_attn,
                                                   dropout=self.dropout)
            masked_embedding = (embedding + attn_embed).masked_fill(mask.unsqueeze(2),
                                                                    value=torch.tensor(-1e9))
            user_embed = torch.squeeze(self.MaxPool(masked_embedding))
        elif self.user_embed_method == "attn2":
            mask_attn = (mask > 0).unsqueeze(1).repeat(1, mask.size(1), 1)
            embedding = self.item_embed(interacted_items)  # [B, max_len, latent_dim]
            attn_embed, _ = self.attention.forward(embedding, embedding, embedding, mask=mask_attn,
                                                   dropout=self.dropout)
            masked_embedding = (embedding + attn_embed).masked_fill(mask.unsqueeze(2),
                                                                    value=torch.tensor(-1e9))
            mask = (1 - mask.to(int)).unsqueeze(2)  # [B, K, 1]
            mean_item_embed = torch.sum(masked_embedding * mask, 1)
            user_embed = self.map(mean_item_embed)
        else:
            user_embed = None
            print("Error, no valid user embedding method specified!")

        return user_embed

    def ItemEmbed(self, item):
        return self.item_embed(item)

    def forward(self, user_id, interacted_items, mask=None):
        user_embed = self.domain_embedding(user_id, interacted_items, mask)
        # pos_embed = self.item_embed(pos)
        # neg_embed = self.item_embed(neg)
        # , pos_embed, neg_embed
        return user_embed


class SingleDomainEmbeddingVQ(SingleDomainEmbedding):

    def __init__(self, n_item=None, latent_dim=50, user_embed="user",
                 pad_ind=0, seq_len=None, code_size=None, pretrain=False,
                 freeze=False, pre_dir=None):
        super(SingleDomainEmbeddingVQ, self).__init__(n_user=None,
                                                      n_item=n_item,
                                                      latent_dim=latent_dim,
                                                      user_embed=user_embed,
                                                      pad_ind=pad_ind,
                                                      seq_len=seq_len,
                                                      pretrain=pretrain,
                                                      freeze=freeze,
                                                      pre_dir=pre_dir)
        self.quantify = Quantification(code_size, int(latent_dim / 8))

    def domain_embedding_vq(self, user_id, interacted_items, mask=None):
        """
        vector quantification of user embedding,
        """
        user_embed = self.domain_embedding(user_id, interacted_items, mask=mask)
        quanti_embed, diff, quanti_ID = self.quantify(user_embed.view(-1, 8, int(self.latent_dim / 8)))
        return quanti_embed, diff, quanti_ID

    def forward(self, user_id, interacted_items, pos, neg, mask=None):
        user_embed, diff, quanti_id = self.domain_embedding_vq(user_id, interacted_items, mask)
        # print(user_embed.size())
        # print(diff.size(), diff)
        # print(quanti_id, quanti_id.size())
        pos_embed = self.item_embed(pos)
        neg_embed = self.item_embed(neg)

        return user_embed.view(-1, self.latent_dim), pos_embed, neg_embed, diff, quanti_id


class Quantification(nn.Module):

    def __init__(self, codebook, latent_dim):
        super(Quantification, self).__init__()
        self.quantize = Quantize(latent_dim, codebook)

    def forward(self, x):
        """
        x: [B, latent_dim]
        return: [B, latent_dim], diff, [B, 1] /[B]
        """
        quanti_embed, diff, quanti_ID = self.quantize(x)

        return quanti_embed, diff, quanti_ID


class Bundler(nn.Module):
    def forward(self, data):
        raise NotImplementedError

    def forward_i(self, data):
        raise NotImplementedError

    def forward_o(self, data):
        raise NotImplementedError


class ItemEmb(Bundler):
    def __init__(self, item_num=20000, factors=100, padding_idx=0):
        super(ItemEmb, self).__init__()
        self.item_num = item_num  # item_num
        self.factors = factors
        self.ivectors = nn.Embedding(self.item_num, self.factors, padding_idx=padding_idx)
        self.ovectors = nn.Embedding(self.item_num, self.factors, padding_idx=padding_idx)
        self.ivectors.weight = nn.Parameter(torch.cat([torch.zeros(1, self.factors),
                                                       torch.FloatTensor(
                                                           self.item_num - 1, self.factors).uniform_(
                                                           -0.5 / self.factors, 0.5 / self.factors)]))
        self.ovectors.weight = nn.Parameter(torch.cat([torch.zeros(1, self.factors),
                                                       torch.FloatTensor(self.item_num - 1,
                                                                         self.factors).uniform_(-0.5 / self.factors,
                                                                                                0.5 / self.factors)]))
        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True

    def forward(self, data):
        return self.forward_i(data)

    def forward_i(self, data):
        # v = torch.LongTensor(data)
        v = data
        v = v.cuda() if self.ivectors.weight.is_cuda else v
        return self.ivectors(v)

    def forward_o(self, data):
        # v = torch.LongTensor(data)
        v = data
        v = v.cuda() if self.ovectors.weight.is_cuda else v
        return self.ovectors(v)


class Item2Vec(nn.Module):
    def __init__(self,
                 item2idx,
                 item_num=20000,
                 factors=100,
                 epochs=20,
                 n_negs=20,
                 padding_idx=0,
                 weights=None,
                 use_cuda=False,
                 early_stop=True):
        """
        Item2Vec Recommender Class with SGNS
        Parameters
        ----------
        item2idx : Dict, {item : index code} relation mapping
        item_num : int, total item number
        factors : int, latent factor number
        epochs : int, training epoch number
        n_negs : int, number of negative samples
        padding_idx : int, pads the output with the embedding vector at padding_idx (initialized to zeros)
        weights : weight value to use
        use_cuda : bool, whether to use CUDA enviroment
        early_stop : bool, whether to activate early stop mechanism
        """
        super(Item2Vec, self).__init__()
        self.item2idx = item2idx
        self.embedding = ItemEmb(item_num, factors, padding_idx)
        self.item_num = item_num
        self.factors = factors
        self.epochs = epochs
        self.n_negs = n_negs
        self.weights = None
        if weights is not None:
            wf = np.power(weights, 0.75)
            wf = wf / wf.sum()
            self.weights = torch.FloatTensor(wf)

        self.use_cuda = use_cuda
        self.early_stop = early_stop

        self.user_vec_dict = {}
        self.item2vec = {}

    def forward(self, iitem, oitems):
        batch_size = iitem.size()[0]
        context_size = oitems.size()[1]
        if self.weights is not None:
            nitems = torch.multinomial(self.weights, batch_size * context_size * self.n_negs,
                                       replacement=True).view(batch_size, -1)
        else:
            nitems = torch.FloatTensor(batch_size,
                                       context_size * self.n_negs).uniform_(0, self.item_num - 1).long()
        ivectors = self.embedding.forward_i(iitem).unsqueeze(2)
        ovectors = self.embedding.forward_o(oitems)
        nvectors = self.embedding.forward_o(nitems).neg()
        oloss = torch.bmm(ovectors, ivectors).squeeze().sigmoid().log().mean(1)
        nloss = torch.bmm(nvectors, ivectors).squeeze().sigmoid().log().view(-1, context_size,
                                                                             self.n_negs).sum(2).mean(1)
        return -(oloss + nloss).mean()

    def fit(self, train_loader):
        item2idx = self.item2idx
        if self.use_cuda and torch.cuda.is_available():
            self.cuda()
        else:
            self.cpu()

        optimizer = optim.Adam(self.parameters())
        last_loss = 0.
        epoch = 0
        # for epoch in range(1, self.epochs + 1):
        current_loss = 0.
        # set process bar display
        pbar = tqdm(train_loader)
        pbar.set_description(f'[Train]')
        step = 0
        for data_batch in pbar:
            iitem, oitems = data_batch['itag'], data_batch['otags']
            loss = self.forward(iitem, oitems)

            if torch.isnan(loss):
                raise ValueError(f'Loss=Nan or Infinity: current settings does not fit the recommender')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())
            current_loss += loss.item()
            step += 1
            if step % 4000 == 0:
                delta_loss = float(current_loss - last_loss) / 4000
                print(f"loss delta {delta_loss}")
                if (abs(delta_loss) < 0.01) and self.early_stop:
                    print('Satisfy early stop mechanism')
                    break
                else:
                    last_loss = current_loss
                epoch += 1
            if epoch > self.epochs:
                break

        idx2vec = self.embedding.ivectors.weight.data.cpu().numpy()

        self.item2vec = {k: idx2vec[v, :] for k, v in item2idx.items()}

    def build_user_vec(self, ur):
        for u in ur.keys():
            self.user_vec_dict[u] = np.array([self.item2vec[i] for i in ur[u]]).mean(axis=0)

    def predict(self, u, i):
        if u in self.user_vec_dict.keys():
            user_vec = self.user_vec_dict[u]
        else:
            return 0.
        item_vec = self.item2vec[i]

        return self._cos_sim(user_vec, item_vec)

    def _cos_sim(self, a, b):
        numerator = np.multiply(a, b).sum()
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if denominator == 0:
            return 0
        else:
            return numerator / denominator


class MulDomainEmbed(nn.Module):
    """
    load user embedding model for all domains
    """
    def __init__(self, domain_embed_class, n_domain, n_tags,
                 latent_dim, user_embed_method, seq_len, device, path=None):
        """
        * domain_embed_class (class name): which method is used to build user embedding at each domain
            TODO: all domain_embed_class should have the same interfaces.
        * path (list): path to pre-trained single domain user embedding models
        * n_domain (int): number of domains
        * n_tags (list of int), number of tags in each domain
        * latent_dim (int), dimensionality of latent embedding
        * user_embed_method (str), indicating the method used for single domain user embedding
        * seq_len (list of int),
        """
        super(MulDomainEmbed, self).__init__()
        model_list = []
        for i in range(n_domain):
            model_list.append(domain_embed_class(n_user=None,
                                                 n_item=n_tags[i] + 1,
                                                 latent_dim=latent_dim,
                                                 user_embed=user_embed_method,
                                                 pad_ind=0,
                                                 seq_len=seq_len[i]))
            # Restore model parameters
            if path is not None:
                if os.path.isfile(path[i]):
                    checkpoint = torch.load(path[i], map_location=torch.device(device))
                    model_list[-1].load_state_dict(checkpoint)
                else:
                    print(f" ==== No pretrained tag embedding for domain {i}")
        self.domain_models = nn.ModuleList(model_list)
        self.n_domain = n_domain

    def ItemEmbed(self, item, domain):
        return self.domain_models[domain].ItemEmbed(item)

    def DomainEmbed(self, tag, mask, domain):
        return self.domain_models[domain](None, tag, mask)

    def forward(self, x, mask):
        """
        x: a batch of user behavior in all domains: [B, 5, seq_len]
        mask: tag level mask, indicating which of the tags in the tag sequence is padded by 0 (pad_index).
        return: user embeddings at each domain: [B, 5, latent_dim]
        """
        domain_embed = []
        for i in range(self.n_domain):
            domain_embed.append(self.domain_models[i](None, x[:, i, :], mask[:, i]))
        return torch.stack(domain_embed, 1)
