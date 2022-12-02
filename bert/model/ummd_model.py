"""
The main User modeling in multiple domain (UMMD) model.
    1. User modeling in a single domain: /bert/model/domain_representation.py
what this model is doing?
    1. input data. user: domain1_data, domain2_data, ...
    2. For each domain, build up a user profiling model in each domain
    3. For a given user, predict their embedding in missing domain.

Training:
    1. pre-train in each individual domain,
    2. Train the UMMD model
        a. fix the parameters in each individual domain?
        b. fine-tune the whole model together?
"""

import torch.nn as nn
from .domain_representation import domain_model
import torch
import numpy as np
from .bert import BERT_m
import sys
import matplotlib.pyplot as plt
from .embedding import PositionalEmbedding


def save_model(model, check_name):
    torch.save(model.state_dict(), check_name)


def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([plt.Line2D([0], [0], color="c", lw=4),
                plt.Line2D([0], [0], color="b", lw=4),
                plt.Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


class UMMD_VQ(nn.Module):

    def __init__(self, n_user=None, n_items=None, codebook_sizes=None, latent_dims=None, user_embed_meth=None,
                 pad_ind=0, user_models=None, device="cpu", hyper_args=None):
        """
        :param n_user: The total number of users (unification over all domains)
        :param n_items: A list of the number of items in each domain.
        :param codebook_sizes: The codebook sizes that we are going to use for each domain
                               (tune in each individual domain)
        :param latent_dims: The latent dimension for user embedding in each domain
                            If they are not equal to each other, add an additional full connect layer.
        :param user_embed_meth: The user embedding method for each domain
                                (in most cases we use consistent user embedding methods across different domains)
        :param pad_ind: padding index. (used to generate masks.)
        """
        super().__init__()
        if user_embed_meth is None:
            user_embed_meth = ["item", "item", "item"]
        if latent_dims is None:
            latent_dims = [50, 50, 50]
        if codebook_sizes is None:
            codebook_sizes = [128, 128, 128]
        if n_items is None:
            print("number of items is not specified.")
            sys.exit()

        self.n_user = n_user
        self.n_items = n_items
        self.codebook_sizes = codebook_sizes
        self.latent_dims = latent_dims
        self.user_embed_method = user_embed_meth
        self.pad_index = pad_ind
        self.num_domains = len(n_items)
        # restore user profiling model in each individual domain.
        self.domain_models = []
        self.device = device
        self.hyper_args = hyper_args
        self.pad_embedding = nn.Embedding(self.num_domains, latent_dims[-1])  # padding embedding for missing domain
        nn.init.normal_(self.pad_embedding.weight, std=0.01)

        # positional embedding.
        self.position = PositionalEmbedding(d_model=self.latent_dims[0])

        if user_models is not None:
            self._load_domain_models(user_models)

        # it is better to set hidden = 32 * attn_heads
        self.bert = BERT_m(hidden=latent_dims[-1], n_layers=self.hyper_args.n_layer,
                           attn_heads=self.hyper_args.attn_heads, dropout=self.hyper_args.dropout)

    def _load_domain_models(self, model_dirs):
        for index, check_point in enumerate(model_dirs):
            # build model
            self.domain_models.append(domain_model(n_user=self.n_user, n_item=self.n_items[index],
                                                   codebook_size=self.codebook_sizes[index],
                                                   latent_dim=self.latent_dims[index],
                                                   user_embed=self.user_embed_method[index],
                                                   pad_ind=self.pad_index,
                                                   seq_len=self.hyper_args.length[index]))
            if self.hyper_args.pretrain == "pretrain":
                # load parameters.
                print(f" ** Loading pretrained model in domain {index}")
                checkpoint = torch.load(check_point, map_location=torch.device(self.device))
                self.domain_models[-1].load_state_dict(checkpoint)  # TODO remember the format.

            # for name, param in self.domain_models[-1].named_parameters():
            #     print(name, param.data, param.requires_grad)

            # to device
            self.domain_models[-1] = self.domain_models[-1].to(self.device)
            # TODO set func to control the train-ability of the single domain embedding model.
            #  (including the use of .detach() in other places.) at the UMMD level.
            # self.domain_models[-1].eval()     # detach

        self.domain_models = nn.ModuleList(self.domain_models)

    def save_model(self):
        """
        models with parameters:
            domain_models
            pad_embedding
            bert
        what to save?
        """
        pass

    def get_domain_embedding(self, user_id, x, mask=None):
        """
        :param mask:    mask which indicate the missing domains for every users.
        :param user_id:   B
        :param x:  [B, n_domain, length], length maybe different in different domains.
        :return:
        Training:
            randomly generate mask and recover those masked embeddings
        Testing:
            Find the mask in the original data, then recover all the embeddings in th missing domains.
        """
        embeddings = []  # original quantified
        embeddings_masked = []  # masked
        embed_ids = []
        embeddings_origin = []  # original

        for domain in range(self.num_domains):
            domain_tensor = torch.from_numpy(np.array([domain])).to(self.device)
            # x_domain = x[:, domain, :]  # [B, seq]
            x_domain = x[domain]
            #             print("Input of each domain", x_domain.size())
            x_domain, diff, id_t, user_embed = self.domain_models[domain].domain_embedding(user_id, x_domain)

            # x_domain_de = x_domain.detach().clone()
            x_domain_de = x_domain

            embeddings_origin.append(user_embed)
            embeddings.append(x_domain_de)
            embed_ids.append(id_t)
            #             print("user embedding size", x_domain.size())
            # masked domain embedding with positional(domain id) embedding
            if mask is not None:
                domain_mask = mask[:, domain]
                # print(domain_mask.size())
                # True False and 1 0 are interchangeable
                domain_mask_index = (domain_mask == 1).nonzero(as_tuple=True)[0]
                if len(domain_mask_index) > 0:
                    x_domain_de[domain_mask_index] = self.pad_embedding(domain_tensor)
            embeddings_masked.append(torch.squeeze(x_domain_de))  # [B, latent_dim]
            # print(f"domain {domain} embedding with shape, {x_domain_de.size()}" )
        # [B, n_domain, latent_dim] pos: [1, n_domain, dim]
        # print(torch.stack(embeddings_masked, 1).size())
        # print(self.position(self.num_domains).size())
        embeddings_masked = torch.stack(embeddings_masked, 1) + self.position(self.num_domains)  # TODO
        # embeddings = torch.stack(embeddings, 1).detach()
        embeddings = torch.stack(embeddings, 1)
        embeddings_org = torch.stack(embeddings_origin, 1)
        embed_ids = torch.stack(embed_ids, 1)
        if self.hyper_args.vq == "True":
            return_embeddings = embeddings
        else:
            return_embeddings = embeddings_org

        return embeddings_masked, return_embeddings, embed_ids

    def recover_embeddings(self, user_id, x, mask):
        """
        :param user_id:
        :param mask: [B, n_domain]; 0, 1 indicate the masked domains.
        :param x: [B, n_domain, ] different domain can have different seq length
        :return: Embeddings: [B, n_domain, latent_dim]
        """
        embeddings_mask, embeddings_org, embed_ids = self.get_domain_embedding(user_id, x, mask)
        mask_attn = (mask > 0).unsqueeze(1).repeat(1, mask.size(1), 1).unsqueeze(1)

        embeddings_recover = self.bert(embeddings_mask, mask_attn.to(self.device))

        return embeddings_recover, embeddings_org, embed_ids

    def forward(self, user_id, x, mask=None):
        """
        :param mask:    masks to show the true(testing) or dummy(train) missing domains of users.
        :param user_id: user_id, may be useful when we have user embedding matrix in each single domain(rarely).
        :param x:       input data, [B, n_dim, user_seq], where the user_seq denote the users' behavior in each domain
                        we support different length of the "user_seq".
        :return:
            embeddings_org  : users' original embedding in each domain (generated by models in each domain)
            embeddings_recover : recovered users' embedding in each domain,
                                train: mask --> recover,
                                test: missing --> filling,
            Note:
                In test the embeddings of the missing of the embeddings_org is meaningless.
        """
        embeddings_recover, embeddings_org, embed_ids = self.recover_embeddings(user_id, x, mask)
        # [B, n_domain, latent_dim], [B, n_domain, embed_ids]

        return embeddings_recover, torch.squeeze(embeddings_org), torch.squeeze(embed_ids)


class UMMD_MLM(nn.Module):
    """
    MLM ontop of the UMMD-type (UMMD-VQ, UMMD-wild) model,
    """

    def __init__(self, ummd_model: UMMD_VQ, n_domain, hidden, codebook_sizes, reuse_book=True):
        """
        reuse_book: where to use the code book as the output layer.
        """
        super().__init__()
        self.ummd_model = ummd_model
        # self.linear = []
        self.vocab_sizes = codebook_sizes
        self.reuse_book = reuse_book

        if not reuse_book:
            self.linear = nn.ModuleList([nn.Linear(hidden, codebook_sizes[i]) for i in range(n_domain)])
            # self.linear.append(nn.Linear(hidden, codebook_sizes[i]))
        # self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, user_id, x, mask):
        """
        :param mask:
        :param user_id:
        :param x:
        :return:
        """
        embeddings_recover, embeddings_org, embed_ids = self.ummd_model(user_id, x, mask)
        logits = []
        # target = []

        for domain_id in range(self.ummd_model.num_domains):
            if self.reuse_book:
                # TODO detach or not
                # logits.append(torch.matmul(torch.squeeze(embeddings_recover[:, domain_id, :]),
                #                            self.ummd_model.domain_models[domain_id].quantize.embed.detach()))
                logits.append(torch.matmul(torch.squeeze(embeddings_recover[:, domain_id, :]),
                                           self.ummd_model.domain_models[domain_id].quantize.embed))
            else:
                logits.append(self.linear[domain_id](embeddings_recover[:, domain_id, :]))
                # [B, vocab_size_i] (i-th domain)

        # target = torch.eye(self.vocab_sizes[domain_id])[embed_ids, :]
        logits = torch.stack(logits, 1)
        # diff = (quantize.detach() - input).pow(2).mean()
        diff = (embeddings_org.detach() - embeddings_recover).pow(2).mean()

        return logits, embed_ids, diff

