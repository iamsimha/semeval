
from __future__ import print_function, division, unicode_literals
import torch
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, pytorch_seq2seq_wrapper
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, pytorch_seq2vec_wrapper
from allennlp.modules import SimilarityFunction
from allennlp.data.vocabulary import Vocabulary
from typing import Dict, List, Iterator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from myallennlp.metrics.emo_metrics import MicroMetrics
from torch import nn
from collections import Counter
from allennlp.nn.util import sort_batch_by_length
from torch.nn.utils.rnn import pack_padded_sequence
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention


import torch.nn.functional as F
import numpy as np
import json
import csv
import argparse

import numpy as np
import emoji

from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_emojis
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

torch.manual_seed(10)

def cross_entropy_loss(predictions, gnd_truth):
    #TODO use mask for computing loss
    loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.3, 0.3, 0.3]))
    return loss(predictions, gnd_truth)

@Model.register("deep-emoji-sentence-lstm")
class EmotionBiLSTM(Model):
    def __init__(self,
                vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.accuracy = MicroMetrics(vocab)
        self.label_index_to_label = self.vocab.get_index_to_token_vocabulary('labels')
        final_concatenated_dimension = 64 * 3
        self.input_layer = torch.nn.Linear(in_features=final_concatenated_dimension, out_features=64)
        self.output_layer = torch.nn.Linear(in_features=64, out_features=vocab.get_vocab_size("labels"))
        self.sigmoid = nn.Sigmoid()
        with open(VOCAB_PATH, 'r') as f:
            self.vocabulary = json.load(f)
            self.st = SentenceTokenizer(self.vocabulary, 20)
        self.model = torchmoji_emojis(PRETRAINED_PATH)


    def tokenize(self, sentences):
        tokenized, _, _ = self.st.tokenize_sentences(sentences)
        return torch.from_numpy(tokenized.astype(np.int))


    
    def forward(self,
                turn1,
                turn2,
                turn3,
                conversation_id: str,
                turns: str,
                labels: torch.Tensor=None):
        #TODO Looku up reverse embedding of padded sequences
        turn1 = [x['turn1'] for x in turn1]
        turn2 = [x['turn2'] for x in turn2]
        turn3 = [x['turn3'] for x in turn3]
        predictions1 = self.model(self.tokenize(turn1))
        predictions2 = self.model(self.tokenize(turn2))
        predictions3 = self.model(self.tokenize(turn3))
        predictions = torch.cat([predictions1, predictions2, predictions3], dim=1)
        input2hidden = self.input_layer(predictions)
        label_logits = self.sigmoid(self.output_layer(input2hidden))

        # self.matrix_attention = self.matrix_attention(encoded_turn1and2, encoded_turn3)
        label_logits = F.softmax(label_logits, dim=1)
        output = {"prediction": [self.label_index_to_label[x] for x in label_logits.argmax(dim=1).numpy()],
                    "ids": [x["ids"] for x in conversation_id],
                    "turns": [x["turns"] for x in turns]}
        if labels is not None:
            #TODO check loss without and with mask
            self.accuracy(label_logits, labels)
            output["loss"] = cross_entropy_loss(label_logits, labels)
        return output
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}