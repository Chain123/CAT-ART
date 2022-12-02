import torch.nn as nn
import torch
from .bert import BERT


class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.next_sentence(x), self.mask_lm(x)


class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class MaskedLanguageModel_share(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden_size, codebook_size):
        """
        """
        super().__init__()
        self.linear = nn.Linear(hidden_size, codebook_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """
        :param x:    The output of the bert model: [B, n_domains, n_dims]
        :return:     logits. The predicted logits for masked domains.
        TODO: 1. Merge code books, and do softmax over all codes vectors. (unify idx over all domains)
              2. Predict each domain independently. (preferred)
        1.
        out = torch.matmul(x, )
        2.
        """
        soft_logits = []
        for domain_id in range(len(self.code_books)):
            soft_logits.append(self.softmax(torch.matmul(torch.squeeze(x[:, domain_id, :]),
                                                         torch.transpose(self.code_books[domain_id], 0, 1))))
            # [B, n_codes]

        return soft_logits   # [n_domains, B, n_codes]
