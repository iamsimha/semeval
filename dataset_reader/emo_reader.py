from typing import Dict, List, Iterator
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField, LabelField, metadata_field

import numpy as np

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

np.random.seed(10)

def replace_emoji(line, emoji_map):
    result = ""
    for c in line:
        if c in emoji_map:
            result += " " + emoji_map[c] + " "
        else:
            result += c
    result = result.strip()
    result.replace("  ", " ")
    return result

def get_text_processor():
    return TextPreProcessor(
        # terms that will be normalized
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
            'time', 'date', 'number'],
        # terms that will be annotated
        annotate={"hashtag", "allcaps", "elongated", "repeated",
            'emphasis', 'censored'},
        fix_html=True,  # fix HTML tokens
        
        # corpus from which the word statistics are going to be used 
        # for word segmentation 
        segmenter="twitter", 
        
        # corpus from which the word statistics are going to be used 
        # for spell correction
        corrector="twitter", 
        
        unpack_hashtags=True,  # perform word segmentation on hashtags
        unpack_contractions=True,  # Unpack contractions (can't -> can not)
        spell_correct_elong=False,  # spell correction for elongated words
        
        # select a tokenizer. You can use SocialTokenizer, or pass your own
        # the tokenizer, should take as input a string and return a list of tokens
        tokenizer=SocialTokenizer(lowercase=True).tokenize,
        
        # list of dictionaries, for replacing tokens extracted from the text,
        # with other expressions. You can pass more than one dictionaries.
        dicts=[emoticons]
    )



@DatasetReader.register("concat-reader")
class EmoConcatDataReader(DatasetReader):
    """
    DatasetReader for Emoition recognition data

        Id, turn1, turn2, turn3, (optional)labels
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, tokens: List[Token], conversation_id, turns: List[str], tag: List[str] = None) -> Instance:
        conversation = TextField(tokens, self.token_indexers)
        conversation_id = metadata_field.MetadataField({'ids': conversation_id})
        turns = metadata_field.MetadataField({'turns': turns})
        fields = {"conversation": conversation, "conversation_id": conversation_id, "turns": turns}
        if tag:
            label_field = LabelField(label=tag)
            fields["labels"] = label_field
        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            label = None
            _ = f.readline()
            for line in f:
                contents = line.strip().split("\t")
                if len(contents) == 5:
                    _id, turn1, turn2, turn3, label = contents
                else:
                    _id, turn1, turn2, turn3 = contents
                turns = [turn1, turn2, turn3]
                sentence = " <eos> ".join(turns)
                sentence = sentence.strip().split()
                if label is not None:
                    yield self.text_to_instance([Token(word) for word in sentence], _id, "\t".join(turns), label)
                else:
                    yield self.text_to_instance([Token(word) for word in sentence],_id, "\t".join(turns))

@DatasetReader.register("bi-sentence-reader")
class EmoBiSentenceDataReader(DatasetReader):
    """
    DatasetReader for Emoition recognition data

        Id, turn1, turn2, turn3, (optional)labels
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.text_processor = get_text_processor().pre_process_doc

    def text_to_instance(self, turn1and2: List[Token], turn3: List[Token], conversation_id, turns: List[str], label: List[str] = None) -> Instance:
        turn1and2  = TextField(turn1and2, self.token_indexers)
        turn3 = TextField(turn3, self.token_indexers)
        conversation_id = metadata_field.MetadataField({'ids': conversation_id})
        turns = metadata_field.MetadataField({'turns': turns})
        fields = {"turn1and2": turn1and2, "turn3": turn3, "conversation_id": conversation_id, "turns": turns}
        if label:
            label_field = LabelField(label=label)
            fields["labels"] = label_field
        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            label = None
            _ = f.readline()
            for line in f:
                contents = line.strip().split("\t")
                if len(contents) == 5:
                    _id, turn1, turn2, turn3, label = contents
                else:
                    _id, turn1, turn2, turn3 = contents
                gnd_turns = "\t".join([turn1, turn2, turn3])
                turn1 = " ".join(self.text_processor(turn1))
                turn2 = " ".join(self.text_processor(turn2))
                turn3 = " ".join(self.text_processor(turn3))
                
                turns = [turn1, turn2]
                turn1and2 = " <eos> ".join(turns)

                # use tokenizer here
                turn1and2 = turn1and2.strip().split()
                turn3 = turn3.strip().split()
                turn1and2_tokens = [Token(word) for word in turn1and2]
                turn3_tokens = [Token(word) for word in turn3]
                if label is not None:
                    yield self.text_to_instance(turn1and2_tokens, turn3_tokens, _id, gnd_turns, label)
                else:
                    yield self.text_to_instance(turn1and2_tokens, turn3_tokens, _id, gnd_turns)

@DatasetReader.register("tri-sentence-reader")
class EmoBiSentenceDataReader(DatasetReader):
    """
    DatasetReader for Emoition recognition data

        Id, turn1, turn2, turn3, (optional)labels
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.text_processor = None

    def text_to_instance(self, turn1:str, turn2:str, turn3: str, conversation_id, turns: List[str], label: List[str] = None) -> Instance:
        conversation_id = metadata_field.MetadataField({'ids': conversation_id})
        turns = metadata_field.MetadataField({'turns': turns})
        turn1 = metadata_field.MetadataField({'turn1': turn1})
        turn2 = metadata_field.MetadataField({'turn2': turn2})
        turn3 = metadata_field.MetadataField({'turn3': turn3})
        fields = {"turn1": turn1, "turn2": turn2, "turn3": turn3, "conversation_id": conversation_id, "turns": turns}
        if label:
            label_field = LabelField(label=label)
            fields["labels"] = label_field
        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            label = None
            _ = f.readline()
            for line in f:
                contents = line.strip().split("\t")
                if len(contents) == 5:
                    _id, turn1, turn2, turn3, label = contents
                else:
                    _id, turn1, turn2, turn3 = contents
                gnd_turns = "\t".join([turn1, turn2, turn3])
                # use tokenizer here
                if label is not None:
                    yield self.text_to_instance(turn1, turn2, turn3, _id, gnd_turns, label)
                else:
                    yield self.text_to_instance(turn1, turn2, turn3, _id, gnd_turns)

 