import torch
from scripts.utils import LstmSeq2SeqEncoder
from torch import nn
from torch.distributions import Categorical


class Generator(nn.Module):

    def __init__(self, latent_dim, vocab_size, start_token, end_token):
        """Generator

        Args:
            latent_dim (int): [description]
            vocab_size (int): vocab size without padding
            start_token ([int]): start token (without padding idx)
            end_token ([int]): end token (without padding idx)
        """

        super().__init__()

        # (-1) we do not need pad token for the generator
        self.vocab_size = vocab_size
        self.start_token = start_token
        self.end_token = end_token

        self.embedding_layer = nn.Embedding(self.vocab_size, latent_dim)

        self.project = nn.Sequential(
            nn.Linear(latent_dim,latent_dim*2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim*2, latent_dim * 2),
            nn.ELU(alpha=0.1),
            nn.Dropout(0.1),
        )
        self.rnn = nn.LSTMCell(latent_dim, latent_dim)
        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, vocab_size - 1)
        )

    def forward(self, z, max_len=100):
        """[summary]

        Args:
            z (torch.Tensor): [description]
            max_len (int, optional): [description]. Defaults to 20.

        Returns:
            dict: x [B, max_len], log_probabilities [B, max_len, vocab], entropies [B,]
        """

        batch_size = z.shape[0]

        # start of sequence
        starts = torch.full(
            size=(batch_size,), fill_value=self.start_token, device=z.device).long()

        # embed_start
        emb = self.embedding_layer(starts)

        x = []
        log_probabilities = []
        entropies = []

        h, c = self.project(z).chunk(2, dim=1)

        for i in range(max_len):

            # new state
            h, c = self.rnn(emb, (h, c))

            # prediction
            logits = self.output_layer(h)

            # create dist
            dist = Categorical(logits=logits)

            # sample
            sample = dist.sample()

            # append prediction
            x.append(sample)

            # append log prob
            log_probabilities.append(dist.log_prob(sample))

            # append entropy
            entropies.append(dist.entropy())

            # new embedding
            emb = self.embedding_layer(sample)

        # stack along sequence dim
        x = torch.stack(x, dim=1)
        log_probabilities = torch.stack(log_probabilities, dim=1)
        entropies = torch.stack(entropies, dim=1)

        # keep only valid lengths (before EOS)
        end_pos = (x == self.end_token).float().argmax(dim=1).cpu()

        # sequence length is end token position + 1
        seq_lengths = end_pos + 1

        # if end_pos = 0 => put seq_length = max_len
        seq_lengths.masked_fill_(seq_lengths == 1, max_len)

        # select up to length
        _x = []
        _log_probabilities = []
        _entropies = []
        for x_i, logp, ent, length in zip(x, log_probabilities, entropies, seq_lengths):
            _x.append(x_i[:length])
            _log_probabilities.append(logp[:length])
            _entropies.append(ent[:length].mean())

        x = torch.nn.utils.rnn.pad_sequence(
            _x, batch_first=True, padding_value=-1)

        x = x + 1  # add padding token

        return {'x': x, 'log_probabilities': _log_probabilities, 'entropies': _entropies}




class RecurrentDiscriminator(nn.Module):

    def __init__(self, hidden_size, vocab_size, start_token, bidirectional=True):
        """Reccurent discriminator

        Args:
            hidden_size (int): model hidden size
            vocab_size (int): vocabulary size
            bidirectional (bool, optional): [description]. Defaults to True.
        """

        super().__init__()

        self.start_token = start_token

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

        self.rnn = LstmSeq2SeqEncoder(
            hidden_size, hidden_size, num_layers=1, bidirectional=bidirectional)

        if bidirectional:
            hidden_size = hidden_size * 2

        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """[summary]

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """

        batch_size, _ = x.size()

        # append start token to the input
        starts = torch.full(
            size=(batch_size, 1), fill_value=self.start_token, device=x.device).long()

        x = torch.cat([starts, x], dim=1)

        mask = x > 0

        # embed input [batch_size, max_len, hidden_size]
        emb = self.embedding(x)

        # contextualize representation
        x = self.rnn(emb, mask)

        # prediction for each sequence
        out = self.fc(x).squeeze(-1)  # [B, max_len]

        return {'out': out[:, 1:], 'mask': mask.float()[:, 1:]}
