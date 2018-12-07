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

NRC_LEXICON_PATH = "/Users/talurj/Documents/Research/MyResearch/SemEval/Emoconnect/lexicons/NRC-Sentiment-Emotion-Lexicons/NRC-Sentiment-Emotion-Lexicons/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
def cross_entropy_loss(predictions, gnd_truth):
    #TODO use mask for computing loss
    loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.3, 0.3, 0.3]))
    return loss(predictions, gnd_truth)

@Model.register("bi-sentence-lstm")
class EmotionBiLSTM(Model):
    def __init__(self, 
                word_embeddings: TextFieldEmbedder,
                encoder1: Seq2SeqEncoder,
                encoder2: Seq2SeqEncoder,
                similarity_function: SimilarityFunction,
                vocab: Vocabulary) -> None:

        super().__init__(vocab)
        self.word_embedding = word_embeddings
        self.enc_turn1and2 = encoder1
        self.enc_turn3 = encoder2
        self.matrix_attention = LegacyMatrixAttention(similarity_function)
        self.accuracy = MicroMetrics(vocab)
        self.label_index_to_label = self.vocab.get_index_to_token_vocabulary('labels')
        final_concatenated_dimension = 9 * self.enc_turn1and2.get_output_dim()
        self.hidden2out = torch.nn.Linear(in_features=final_concatenated_dimension, out_features=vocab.get_vocab_size("labels"))
        self.lexicon_embedding = LexiconEmbedder(NRC_LEXICON_PATH, self.vocab)


    def forward(self,
                turn1and2: Dict[str, torch.Tensor],
                turn3: Dict[str, torch.Tensor],
                conversation_id: str,
                turns: str,
                labels: torch.Tensor=None):
        #TODO Looku up reverse embedding of padded sequences
        turn1and2_mask = get_text_field_mask(turn1and2)
        turn3_mask = get_text_field_mask(turn3)
        turn1and2_word_embeddings = self.word_embedding(turn1and2)
        turn3_word_embeddings = self.word_embedding(turn3)
        turn1and2_sentiment_embeddings = self.lexicon_embedding(turn1and2["tokens"])
        turn3_sentiment_embeddings = self.lexicon_embedding(turn3["tokens"])
        turn1and2_embeddings = torch.cat([turn1and2_word_embeddings, turn1and2_sentiment_embeddings], dim=2)
        turn3_embeddings = torch.cat([turn3_word_embeddings, turn3_sentiment_embeddings], dim=2)


        encoded_turn1and2 = self.enc_turn1and2(turn1and2_embeddings, turn1and2_mask)
        encoded_turn3 = self.enc_turn3(turn3_embeddings, turn3_mask)
        # Different Pooling strategies
        turn1andturn2_max = encoded_turn1and2.max(dim=1)[0]
        encoded_turn3_max = encoded_turn3.max(dim=1)[0]
        max_product = turn1andturn2_max * encoded_turn3_max

        turn1andturn2_min = encoded_turn1and2.min(dim=1)[0]
        encoded_turn3_min = encoded_turn3.min(dim=1)[0]
        min_product = turn1andturn2_min * encoded_turn3_min

        turn1andturn2_mean = torch.mean(encoded_turn1and2, 1)
        encoded_turn3_mean = torch.mean(encoded_turn3, 1)
        mean_product = turn1andturn2_mean * encoded_turn3_mean

        concatenated_vector = torch.cat([turn1andturn2_max, encoded_turn3_max, max_product,
                                        turn1andturn2_min, encoded_turn3_min, min_product,
                                        turn1andturn2_mean, encoded_turn3_mean, mean_product], dim=1)
        # self.matrix_attention = self.matrix_attention(encoded_turn1and2, encoded_turn3)
        label_logits = self.hidden2out(concatenated_vector)
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