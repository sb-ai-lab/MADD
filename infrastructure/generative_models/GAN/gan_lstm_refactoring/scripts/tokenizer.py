import torch


class Tokenizer(object):

    def __init__(self, data):

        unique_char = list(set(''.join(data))) + ['<eos>'] + ['<sos>']

        self.mapping = {'<pad>': 0}

        for i, c in enumerate(unique_char, start=1):
            self.mapping[c] = i

        self.inv_mapping = {v: k for k, v in self.mapping.items()}

        self.start_token = self.mapping['<sos>']

        self.end_token = self.mapping['<eos>']

        self.vocab_size = len(self.mapping.keys())

    def encode_smile(self, mol, add_eos=True):

        out = [self.mapping[i] for i in mol]

        if add_eos:
            out = out + [self.end_token]

        return torch.LongTensor(out)

    def batch_tokenize(self, batch):

        out = map(lambda x: self.encode_smile(x), batch)

        return torch.nn.utils.rnn.pad_sequence(list(out), batch_first=True)
