import selfies
from SmilesPE.pretokenizer import atomwise_tokenizer, kmer_tokenizer
import spacy
import re


class moltokenize(object):
    def tokenizer(self, sentence):
        return [tok for tok in atomwise_tokenizer(sentence) if tok != " "]
