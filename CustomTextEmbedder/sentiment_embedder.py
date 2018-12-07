import numpy as np
import torch
from torch.nn import Embedding
class LexiconEmbedder(object):
    def __init__(self, fname, vocab):
        self.index_to_token = vocab.get_index_to_token_vocabulary("tokens")
        self.token_to_index = vocab.get_token_to_index_vocabulary("tokens")
        self.fname = fname
        self.embedding = self._read_lexicon_file()

    def _read_lexicon_file(self):
        def read_unique_emotions(fname):
            emotions = set()
            with open(fname) as f:
                for line in f:
                    if len(line.split("\t")) == 3:
                        emotions.add(line.split("\t")[1])
            return emotions
        unique_emotions = read_unique_emotions(self.fname)
        emotion_to_id = {emotion:i for i, emotion in enumerate(unique_emotions)}
        max_tokens = max(self.index_to_token.keys())
        embedding = np.zeros((max_tokens + 1, len(emotion_to_id)))
        with open(self.fname) as f:
            for line in f:
                if len(line.split("\t")) == 3:
                    current_word, emotion, association = line.split("\t")
                    if current_word in self.token_to_index:
                        embedding[self.token_to_index[current_word], emotion_to_id[emotion]] = int(association.strip())
        return Embedding.from_pretrained(torch.tensor(embedding.astype(np.float32)))
    def __call__(self, inputs):
        return self.embedding(inputs)
        
        