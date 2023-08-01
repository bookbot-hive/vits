import torch
import torch.nn as nn
from torch.nn import functional as F

from models import TextEncoder

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.dense = nn.Linear(hidden_channels, hidden_channels)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(hidden_channels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, n_vocab, hidden_channels):
        super().__init__()
        self.transform = BertPredictionHeadTransform(hidden_channels)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_channels, n_vocab, bias=False)

        self.bias = nn.Parameter(torch.zeros(n_vocab))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, n_vocab, hidden_channels):
        super().__init__()
        self.predictions = BertLMPredictionHead(n_vocab, hidden_channels)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertForMaskedLM(nn.Module):
    def __init__(
        self,
        n_vocab,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.enc_p = TextEncoder(
            n_vocab,
            out_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )

        self.cls = BertOnlyMLMHead(n_vocab, hidden_channels)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, x, x_lengths, labels=None):
        sequence_output, *_ = self.enc_p(x, x_lengths)  # [b, h, t]
        sequence_output = torch.transpose(sequence_output, 1, -1)  # [b, t, h]
        prediction_scores = self.cls(sequence_output)  # [b, t, v]

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = self.loss_fct(
                prediction_scores.view(-1, self.n_vocab), labels.view(-1)
            )

        return masked_lm_loss, prediction_scores, sequence_output
