from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor


@Predictor.register('emotion_predictor')
class EmotionPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self.i = 0

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence" : sentence})

    def dump_line(self, outputs):
        line = ""
        if self.i == 0:
            self.i += 1
            line = "\t".join(['id', 'turn1', 'turn2', 'turn3', 'label']) + "\n"
        line += "\t".join([outputs["ids"], outputs["turns"], outputs["prediction"]]) + "\n"
        return line

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"words"`` to the output.
        """
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(sentence)