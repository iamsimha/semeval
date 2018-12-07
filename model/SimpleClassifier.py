import torch
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, pytorch_seq2seq_wrapper
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, pytorch_seq2vec_wrapper
from allennlp.data.vocabulary import Vocabulary
from typing import Dict, List, Iterator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from myallennlp.metrics.emo_metrics import MicroMetrics
from torch import nn
from collections import Counter
from allennlp.nn.util import sort_batch_by_length
from torch.nn.utils.rnn import pack_padded_sequence

import torch.nn.functional as F
import numpy as np
torch.manual_seed(10)

def cross_entropy_loss(predictions, gnd_truth):
    loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.3, 0.3, 0.3]))
    return loss(predictions, gnd_truth)

@Model.register("simple-lstm")
class EmotionLSTM(Model):
    def __init__(self, word_embeddings: TextFieldEmbedder, encoder: Seq2VecEncoder, vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embedding = word_embeddings
        self.encoder = encoder
        self.hidden2out = torch.nn.Linear(in_features=encoder.get_output_dim(), out_features=vocab.get_vocab_size("labels"))
        self.accuracy = MicroMetrics(vocab)
        self.lstm = nn.LSTM(input_size=word_embeddings.get_output_dim(), hidden_size=128, num_layers=1, batch_first=True)
        self.label_index_to_label = self.vocab.get_index_to_token_vocabulary('labels')

    
    def forward(self, conversation: Dict[str, torch.Tensor], conversation_id: str, turns: str, labels: torch.Tensor=None):
        #TODO Looku up reverse embedding of padded sequences
        mask = get_text_field_mask(conversation)
        embeddings = self.word_embedding(conversation)
        encoder_out = self.encoder(embeddings, mask)
        label_logits = self.hidden2out(encoder_out)
        label_logits = F.softmax(label_logits, dim=1)
        output = {"prediction": [self.label_index_to_label[x] for x in label_logits.argmax(dim=1).numpy()],
                    "ids": [x["ids"] for x in conversation_id],
                    "turns": [x["turns"] for x in turns]}
        if labels is not None:
            self.accuracy(label_logits, labels)
            output["loss"] = cross_entropy_loss(label_logits, labels)
        return output
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}