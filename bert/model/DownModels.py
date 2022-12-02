"""
Down stream models Based on pre-trained UMMD

How to fine-tune:
1. Directly use the UMMD recover output for supervised fine-tuning:
    self.finetune_stage = "ummd"
2. Use the original embedding and the UMMD output to replace missing domain for supervised task fine-tune:
    self.finetune_stage = "recover"
Try to combine both.

Which strategy should be used in the inference stage?
    Try both

"""
import torch

from .ummd_model import UMMD_MLM
import torch.nn as nn
import torch.nn.functional as F


class Gender(nn.Module):
    """
    Gender prediction
    """
    def __init__(self, ummd_model: UMMD_MLM, MidDims, OutDim, params):
        """
        """
        super(Gender, self).__init__()
        self.UMMD_model = ummd_model
        self.MidDims = MidDims
        self.params = params
        self.tune_strategy = params.gender_method
        # build model.
        model_list = []
        for i in range(len(MidDims) - 1):
            model_list.append(nn.Linear(MidDims[i], MidDims[i + 1]))
            model_list.append(nn.ReLU())
            model_list.append(nn.Dropout(0.2))
        model_list.append(nn.Linear(MidDims[-1], OutDim))    # output layer
        self.ForwardBlocks = nn.ModuleList(model_list)

    def SetTuneStrategy(self, strategy):
        self.tune_strategy = strategy

    def forward(self, data, mask):
        embeddings_recover, embeddings_org, embed_ids = self.UMMD_model.ummd_model(None, data, mask)

        # *** gender prediction
        # [B, 5, latent_dim]
        # if self.params.gender_embedding == "ummd":
        if self.tune_strategy == "origin":
            x = embeddings_org.view(embeddings_org.size()[0], -1)   # MidDims[0] = 5 * latent_dim
            # for layer in self.ForwardBlocks:
            #     x = layer(x)
        elif self.tune_strategy == "recover":   #
            x = embeddings_recover.view(embeddings_recover.size()[0], -1)   # MidDims[0] = 5 * latent_dim
        else:   # combine  TODO
            x = embeddings_org.view(embeddings_org.size()[0], -1)   # MidDims[0] = 5 * latent_dim
            print("Wrong down stream strategy!")

        for layer in self.ForwardBlocks:
            x = layer(x)

        # *** vq logits
        logits = []
        for domain_id in range(self.UMMD_model.ummd_model.num_domains):
            if self.UMMD_model.reuse_book:  # reuse the codebook weights as the weight of the last layer.
                logits.append(torch.matmul(torch.squeeze(embeddings_recover[:, domain_id, :]),
                                           self.UMMD_model.ummd_model.domain_models[domain_id].quantize.embed))
            else:
                logits.append(self.UMMD_model.linear[domain_id](embeddings_recover[:, domain_id, :]))

        # target = torch.eye(self.vocab_sizes[domain_id])[embed_ids, :]
        logits = torch.stack(logits, 1)
        diff = (embeddings_org.detach() - embeddings_recover).pow(2).mean()
        return x, logits, embed_ids, diff     # [B, OutDim]
