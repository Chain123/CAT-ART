# -*- coding: utf-8 -*-

"""
Models.
"""

import torch
import torch.nn as nn
import numpy as np
import loss_functions as loss_func
from bert import MultiHeadedAttention


######################################
# Single domain Recommendation module #
######################################
class PureMF(nn.Module):
    def __init__(self,
                 config: dict):
        super(PureMF, self).__init__()
        self.num_users = config['n_users']
        self.num_items = config['n_items']
        self.latent_dim = config['latent_dim']
        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users + 1, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items + 1, embedding_dim=self.latent_dim)

    def getUsersRating(self, users):
        # user preference score for all items
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def forward(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb = self.embedding_item(pos.long())
        neg_emb = self.embedding_item(neg.long())
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        return loss, reg_loss

    def bpr_loss(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb * items_emb, dim=1)
        return self.f(scores)

"""
class DomainInvariant(nn.Module):

    def __init__(self, config):
        super(DomainInvariant, self).__init__()
        # user embedding tables for all domains
        user_embed_module = []
        for i in range(len(config['n_items'])):
            model_dict = torch.load(config['single_dirs'][i], map_location=config['device'])
            sizes = model_dict['embedding_user.weight'].size()
            embed = torch.nn.Embedding(sizes[0], sizes[1], _weight=model_dict['embedding_user.weight'])
            # reload from single domain model

        # model to extract domain-invariant latent-feature

    def forward(self, *input):
        pass
"""


##############################
#  Cross-domain methods
# Transfer when embeddings in other domains are always available. --> deal with negative transfer problem.
# expected results: baseline models have NT problem, our model can solve this problem
# base1: target + adapt(cat(sources))  -"DONE"
# base2: adapt(cat(target + sources))  -"TOBE TESTED"
# base3: Our best: [all_source] --> MLP --> attention with target_embed + target_embed  --"TO BE TESTED"
# base4: Our best: [all_source] --> MLP --> attention with target_embed + target_embed + domain-independent embedding
# base4_v: with only domain-independent embedding      --"IMPLEMENTING"

# Transfer when embeddings in other domains are sparse --> generating embeddings in multi-target scenario
# Expected results: 1. show sparse reduce transfer results in phase one
# (if possible further demonstrate the robustness of our transfer method.)
# base5: pad vector/0 for missing domain(s). + our transfer method
# base6: pad embeddings generated from our generator (auto-encoder) + our method. In this case, the auto-encoder
# is further trained with adversarial net? or VQ.  "Accurately generation"
##############################


class MultiRecommendBase(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.target_domain = config['target_domain']
        self.latent_dim = config['latent_dim']
        self.num_domains = len(config['single_dirs'])
        # load all the user embeddings in all domains
        user_embedding_list = []
        for i in range(self.num_domains):
            print(f"Loading pre-trained user embedding for domain {i}")
            model_dict = torch.load(config['single_dirs'][i], map_location=config['device'])
            # user embedding table sizes
            sizes = model_dict['embedding_user.weight'].size()
            user_embedding_list.append(torch.nn.Embedding(sizes[0], sizes[1],
                                                          _weight=model_dict['embedding_user.weight']))
        self.user_embedding_list = nn.ModuleList(user_embedding_list)
        # item embedding
        model_dict = torch.load(config['single_dirs'][config['target_domain']], map_location=config['device'])
        sizes = model_dict['embedding_item.weight'].size()
        self.embedding_item = torch.nn.Embedding(sizes[0], sizes[1],
                                                 _weight=model_dict['embedding_item.weight'])
        self.fix_user = config['fix_user']
        self.fix_item = config['fix_item']
        self.f = nn.Sigmoid()

    def getUserEmbed(self, uid, mask=None):
        if self.fix_user:
            with torch.no_grad():
                target_embedding = self.user_embedding_list[self.target_domain](uid)  # [B, latent_dim]
        else:
            target_embedding = self.user_embedding_list[self.target_domain](uid)  # [B, latent_dim]
        source_embeddings = []
        with torch.no_grad():
            for ind in range(len(self.user_embedding_list)):
                if ind != self.target_domain:
                    source_embeddings.append(self.user_embedding_list[ind](uid))
        source_embeddings = torch.stack(source_embeddings, 1)  # [B, N-1, dim]
        user_embed = target_embedding + self.merge_layer(source_embeddings)
        return user_embed

    def getUsersRating(self, users):
        # user preference score for all items
        users = users.long()
        users_emb = self.getUserEmbed(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def forward(self, users, pos, neg):
        users_emb = self.getUserEmbed(users.long())
        if self.fix_item:
            with torch.no_grad():
                pos_emb = self.embedding_item(pos.long())
                neg_emb = self.embedding_item(neg.long())
        else:
            pos_emb = self.embedding_item(pos.long())
            neg_emb = self.embedding_item(neg.long())
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        return loss, reg_loss

    def bpr_loss(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.getUserEmbed(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb * items_emb, dim=1)
        return self.f(scores)


class MultiRecommendBase1(MultiRecommendBase):
    """
    score = (target + f(all_source_embed))  * item_embed
    """

    def __init__(self, config):
        super().__init__(config)
        self.merge_layer = nn.Sequential(nn.Flatten(),
                                         nn.Linear(config['latent_dim'] * (self.num_domains - 1), config['latent_dim']))


class MultiRecommendBase2(MultiRecommendBase):

    def __init__(self, config):
        super().__init__(config)
        self.merge_layer = nn.Sequential(nn.Flatten(),
                                         nn.Linear(config['latent_dim'] * self.num_domains, config['latent_dim']))

    def getUserEmbed(self, uid, mask=None):
        source_embeddings = []
        for ind in range(len(self.user_embedding_list)):
            if ind == self.target_domain:
                if self.fix_user:
                    with torch.no_grad():
                        source_embeddings.append(self.user_embedding_list[ind](uid))
                else:
                    source_embeddings.append(self.user_embedding_list[ind](uid))
            else:
                with torch.no_grad():
                    source_embeddings.append(self.user_embedding_list[ind](uid))
        source_embeddings = torch.stack(source_embeddings, 1)  # [B, N-1, dim]
        user_embed = self.merge_layer(source_embeddings)
        return user_embed + source_embeddings[:, self.target_domain, :]


class MultiRecommendBase3(MultiRecommendBase):
    """ Our best for domain specific transfer
    [all_source] --> MLP --> attention with target_embed + target_embed --> final user_embed for similarity
    """

    def __init__(self, config):
        super().__init__(config)
        adapt_list = []
        for i in range(5):
            adapt_list.append(nn.Linear(self.latent_dim, self.latent_dim))
        self.domain_adapt = nn.ModuleList(adapt_list)
        # valina attention
        self.attention = MultiHeadedAttention(h=1, d_model=self.latent_dim)

    def getUserEmbed(self, users, mask=None):
        adapt_user_embed = []
        target_embed = []
        for i in range(5):
            if i != self.target_domain:
                # with torch.no_grad():   # TODO
                user_embed_domain = self.user_embedding_list[i](users)
                adapt_user_embed.append(self.domain_adapt[i](user_embed_domain))  # [B, latent_dim]
            else:
                if self.fix_user:
                    with torch.no_grad():
                        target_embed.append(self.user_embedding_list[i](users))  # [B, latent_dim]
                else:
                    target_embed.append(self.user_embedding_list[i](users))  # [B, latent_dim]
                # adapt_user_embed.append(self.domain_adapt[i](self.user_embedding_list[i](users))) # [B, latent_dim]
        adapt_user_embed = torch.stack(adapt_user_embed, 1)
        target_embed = torch.stack(target_embed, 1)
        user_embed = self.attention(target_embed, adapt_user_embed, adapt_user_embed)
        return torch.squeeze(user_embed + target_embed)


class MultiRecommendBase4(MultiRecommendBase):
    """
    Only transfer domain-independent feature.
    auto-encoder? +
    contrastive learning
    To get Domain-independent, make domains
    """

    def __init__(self, config):
        super().__init__(config)
        self.Encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config['latent_dim'] * self.num_domains, config['latent_dim'] * 5),
            nn.PReLU(),
            nn.Linear(config['latent_dim'] * 5, config['latent_dim'] * 3),
            nn.PReLU(),
            nn.Linear(config['latent_dim'] * 3, config['latent_dim']))
        # domain-independent embedding
        self.Decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config['latent_dim'], config['latent_dim'] * 3),
            nn.PReLU(),
            nn.Linear(config['latent_dim'] * 3, config['latent_dim'] * 5),
            nn.PReLU(),
            nn.Linear(config['latent_dim'] * 5, config['latent_dim'] * self.num_domains))
        self.pad_embedding = nn.Embedding(num_embeddings=1, embedding_dim=self.latent_dim)
        # domain-specific representations
        adapt_list = []
        for i in range(5):
            adapt_list.append(nn.Linear(self.latent_dim, self.latent_dim))
        self.domain_adapt = nn.ModuleList(adapt_list)
        # valina attention
        self.attention = MultiHeadedAttention(h=1, d_model=self.latent_dim)
        # fixed part
        user_embedding_list_fix = []
        for i in range(self.num_domains):
            print(f"Loading pre-trained user embedding for domain {i}")
            model_dict = torch.load(config['single_dirs'][i], map_location=config['device'])
            # user embedding table sizes
            sizes = model_dict['embedding_user.weight'].size()
            user_embedding_list_fix.append(torch.nn.Embedding(sizes[0], sizes[1],
                                                              _weight=model_dict['embedding_user.weight']))
        self.user_embedding_list_fix = nn.ModuleList(user_embedding_list_fix)
        self.general_embedding_adapt = nn.Linear(self.latent_dim, self.latent_dim)

        self.device = config['device']
        self.similarity = loss_func.Similarity(temp=0.1)
        self.mse = nn.MSELoss(reduce='mean')
        self.final_embed = config['final_embed']

    def getOrgEmbed(self, users, mask=None):
        with torch.no_grad():
            embeds = []
            for i in range(5):
                # embeds.append(self.user_embedding_list_fix[i](users.to('cpu')))
                embeds.append(self.user_embedding_list_fix[i](users))
        embeds = torch.stack(embeds, 1)
        if mask is not None:  # for sparse user case
            mask_index = (mask == 1).nonzero(as_tuple=True)
            embeds[mask_index[0], mask_index[1], :] = self.pad_embedding(torch.zeros(1).to(self.device))
        return embeds.to(self.device)

    def GeneralForward(self, users, mask):
        """
        Args:
            users: user IDs             [B]
            mask: domain-level masking, [B, 5]
        Need pre-training.
        """
        # TODO: target-domain should not be masked?
        # Get embeddings
        embeds = self.getOrgEmbed(users)
        # mask
        mask_index = (mask == 1).nonzero(as_tuple=True)
        embeds_mask = embeds.clone()
        embeds_mask[mask_index[0], mask_index[1], :] = self.pad_embedding(torch.zeros(1).long().to(self.device))
        # Go-through Encoder
        embeds_hide = self.Encoder(embeds)  # [N, dim]
        embeds_mask_hid = self.Encoder(embeds_mask)  # [N, dim]
        # Contrastive loss
        contrastive_loss = loss_func.contrastive_loss_sample_N(embeds_hide, embeds_mask_hid,
                                                               mask, self.similarity, device=self.device)
        # Reconstruction loss
        embeds_rec = self.Decoder(embeds_hide)
        embeds_mask_rec = self.Decoder(embeds_mask_hid)
        embeds = embeds.view(-1, embeds_rec.size(1))
        loss_rec1 = self.mse(embeds_rec, embeds)
        loss_rec2 = self.mse(embeds_mask_rec, embeds)
        return contrastive_loss, loss_rec1, loss_rec2

    def getUserEmbed(self, uid, mask=None):
        embeds = self.getOrgEmbed(uid)
        with torch.no_grad():
            domain_independent_embed = self.Encoder(embeds)
        domain_independent_embed = self.general_embedding_adapt(domain_independent_embed)
        # merge domain-specific embeddings
        adapt_user_embed = []
        target_embed = []
        for i in range(5):
            if i != self.target_domain:
                # with torch.no_grad():
                user_embed_domain = self.user_embedding_list[i](uid)
                adapt_user_embed.append(self.domain_adapt[i](user_embed_domain))  # [B, latent_dim]
            else:
                if self.fix_user:
                    with torch.no_grad():
                        target_embed.append(self.user_embedding_list[i](uid))  # [B, latent_dim]
                else:
                    target_embed.append(self.user_embedding_list[i](uid))  # [B, latent_dim]
                # adapt_user_embed.append(self.domain_adapt[i](self.user_embedding_list[i](users))) # [B, latent_dim]
        # TODO: add domain-independent.
        # if self.final_embed == "both":
        #     adapt_user_embed.append(domain_independent_embed)
        # if self.final_embed == "d_sp":
        #     pass
        # if self.final_embed == "d_in":
        #     adapt_user_embed = [domain_independent_embed]

        adapt_user_embed = torch.stack(adapt_user_embed, 1)
        target_embed = torch.stack(target_embed, 1)
        domain_specific_embed = self.attention(target_embed, adapt_user_embed, adapt_user_embed)
        target_embed, domain_specific_embed = torch.squeeze(target_embed), torch.squeeze(domain_specific_embed)
        if self.final_embed == "both":
            return torch.squeeze(target_embed + domain_specific_embed + domain_independent_embed)
        if self.final_embed == "d_sp":
            return torch.squeeze(target_embed + domain_specific_embed)
        if self.final_embed == "d_in":
            return torch.squeeze(target_embed + domain_independent_embed)


class MultiRecommendBase5(MultiRecommendBase):
    """ Our Final method

    Transfer both the domain specific and domain-independent embedding but in a different manner.
    + generate embeddings for missing domain(s)

    """

    def __init__(self, config):
        super().__init__(config)

        adapt_list = []
        for i in range(5):
            adapt_list.append(nn.Linear(self.latent_dim, self.latent_dim))
        self.domain_adapt = nn.ModuleList(adapt_list)
        # valina attention
        self.attention = MultiHeadedAttention(h=1, d_model=self.latent_dim)

        # domain-independent embedding
        self.high_level_embed_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config['latent_dim'] * self.num_domains, config['latent_dim'] * 6),
            nn.PReLU(),
            nn.Linear(config['latent_dim'] * self.num_domains, config['latent_dim'] * 6)
        )
