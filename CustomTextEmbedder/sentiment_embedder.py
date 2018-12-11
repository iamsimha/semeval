import numpy as np
import torch
from torch.nn import Embedding
class LexiconEmbedder(object):
    def __init__(self, fdict, vocab):
        self.index_to_token = vocab.get_index_to_token_vocabulary("tokens")
        self.token_to_index = vocab.get_token_to_index_vocabulary("tokens")
        self.emotion_embedding = self.get_emotion_embedding(fdict['emotion_lexicon'])
        self.affect_embedding = self.get_affect_intensity_embedding(fdict['affect_lexicon'])
    @staticmethod
    def read_unique_emotions(fname, column_number):
        """
        fname: filename to read lexicon from
        column_number: column on which the emotion is present
        """
        emotions = set()
        with open(fname) as f:
            for line in f:
                if len(line.split("\t")) == 3:
                    emotions.add(line.split("\t")[column_number])
        return emotions
    def get_emotion_embedding(self, fname):
        unique_emotions = self.read_unique_emotions(fname, 1)
        emotion_to_id = {emotion:i for i, emotion in enumerate(unique_emotions)}
        max_tokens = max(self.index_to_token.keys())
        embedding = np.zeros((max_tokens + 1, len(emotion_to_id)))
        all_words = set()
        with open(fname) as f:
            for line in f:
                if len(line.split("\t")) == 3:
                    current_word, emotion, association = line.split("\t")
                    if current_word in self.token_to_index:
                        embedding[self.token_to_index[current_word], emotion_to_id[emotion]] = int(association.strip())
                    all_words.add(current_word)
        
        with open("emotion_not_found.txt", "w") as f:
            for word in set(self.token_to_index.keys()).difference(all_words):
                f.write(word + "\n")
        return Embedding.from_pretrained(torch.tensor(embedding.astype(np.float32)))
    def get_affect_intensity_embedding(self, fname):
        unique_emotions = self.read_unique_emotions(fname, 2)
        emotion_to_id = {emotion:i for i, emotion in enumerate(unique_emotions)}
        max_tokens = max(self.index_to_token.keys())
        embedding = np.zeros((max_tokens + 1, len(emotion_to_id)))
        with open(fname) as f:
            _ = f.readline()
            for line in f:
                if len(line.split("\t")) == 3:
                    word, score, emotion = line.split("\t")
                    if word in self.token_to_index:
                        embedding[self.token_to_index[word], emotion_to_id[emotion]] = float(score.strip())
        return Embedding.from_pretrained(torch.tensor(embedding.astype(np.float32)))
    def __call__(self, inputs):
        return torch.cat([self.emotion_embedding(inputs), self.affect_embedding(inputs)], dim=2)
        
        