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
from myallennlp.CustomTextEmbedder.sentiment_embedder import LexiconEmbedder
import torch.nn.functional as F
import numpy as np

torch.manual_seed(10)

LEXICON_PATH = {"emotion_lexicon": "/Users/talurj/Documents/Research/MyResearch/SemEval/Emoconnect/lexicons/NRC-Sentiment-Emotion-Lexicons/NRC-Sentiment-Emotion-Lexicons/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
"affect_lexicon": "/Users/talurj/Documents/Research/MyResearch/SemEval/Emoconnect/lexicons/NRC-Sentiment-Emotion-Lexicons/NRC-Sentiment-Emotion-Lexicons/NRC-Affect-Intensity-Lexicon/NRC-AffectIntensity-Lexicon.txt"}

def cross_entropy_loss(predictions, gnd_truth):
    #TODO use mask for computing loss
    loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.3, 0.3, 0.3]))
    return loss(predictions, gnd_truth)

@Model.register("simple-bi-lstm")
class EmotionSimpleLSTM(Model):
    def __init__(self, 
                word_embeddings: TextFieldEmbedder,
                encoder: Seq2VecEncoder,
                vocab: Vocabulary) -> None:

        super().__init__(vocab)
        self.word_embedding = word_embeddings
        self.encoder = encoder
        self.accuracy = MicroMetrics(vocab)
        self.label_index_to_label = self.vocab.get_index_to_token_vocabulary('labels')
        self.hidden2out = torch.nn.Linear(in_features=self.encoder.get_output_dim(), out_features=vocab.get_vocab_size("labels"))
        self.lexicon_embedding = LexiconEmbedder(LEXICON_PATH, self.vocab)


    def forward(self,
                all_turns: Dict[str, torch.Tensor],
                conversation_id: str,
                turns: str,
                labels: torch.Tensor=None):
        #TODO Looku up reverse embedding of padded sequences
        all_turns_mask = get_text_field_mask(all_turns)
        all_turns_word_embeddings = self.word_embedding(all_turns)
        all_turns_sentiment_embeddings = self.lexicon_embedding(all_turns["tokens"])
        all_turns_embeddings = torch.cat([all_turns_word_embeddings, all_turns_sentiment_embeddings], dim=2)
        encoded_all_turns = self.encoder(all_turns_embeddings, all_turns_mask)
        label_logits = self.hidden2out(encoded_all_turns)
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