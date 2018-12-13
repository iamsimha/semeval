import numpy as np
from allennlp.training.metrics.metric import Metric
from keras.utils import to_categorical

import warnings
np.random.seed(10)
@Metric.register("micro_emo_f1")
class MicroMetrics(Metric):
    def __init__(self, vocab):
        self.token_mapping = vocab.get_token_to_index_vocabulary('labels')
        self.predictions = np.empty((0, 4), dtype=float)
        self.ground_truth = np.empty((0), dtype=int)
    def __call__(self, predictions, ground_truth):
        self.predictions = np.vstack([self.predictions, predictions.detach().numpy()])
        self.ground_truth = np.hstack([self.ground_truth, ground_truth.detach().numpy()])
    def microf1(self, predictions, ground_truth):
        others_index = self.token_mapping["others"]
        num_other_sample = np.sum(ground_truth == others_index)
        total_sample_size = num_other_sample / 0.88
        remaining_samples_per_class = int((total_sample_size - num_other_sample) / 3)
        indices_for_prediction = np.nonzero(ground_truth == others_index)[0]
        for i in range(4):
            if i == others_index:
                continue
            i_class_labels = np.nonzero(ground_truth == i)[0]
            if i_class_labels.shape[0] > 0:
                i_class_sampled_labels = np.random.choice(i_class_labels, remaining_samples_per_class)
                indices_for_prediction = np.concatenate([indices_for_prediction, i_class_sampled_labels], axis =0)
        predictions_sampled = predictions[indices_for_prediction, :]
        discrete_predictions = np.zeros_like(predictions_sampled)
        pred_max_indices = predictions_sampled.argmax(axis=1)
        discrete_predictions[np.arange(discrete_predictions.shape[0]), pred_max_indices] = 1
        ground_truth = ground_truth[indices_for_prediction]
        discrete_ground_truth = np.zeros((ground_truth.shape[0], 4))
        discrete_ground_truth[np.arange(discrete_ground_truth.shape[0]), ground_truth] = 1
        ground_truth = discrete_ground_truth
        true_positives = np.sum(discrete_predictions * ground_truth, axis=0)
        false_positives = np.sum(np.clip(discrete_predictions - ground_truth, 0, 1), axis=0)
        false_negatives = np.sum(np.clip(ground_truth - discrete_predictions, 0, 1), axis=0)
        true_positives = np.hstack([true_positives[:others_index], true_positives[others_index + 1:]])
        false_positives = np.hstack([false_positives[:others_index], false_positives[others_index + 1:]])
        false_negatives = np.hstack([false_negatives[:others_index], false_negatives[others_index + 1:]])
        

        return true_positives, false_positives, false_negatives

    def get_metric(self, reset):
        true_positives, false_positives, false_negatives = self.microf1(self.predictions, self.ground_truth)
        prec_denominator = (np.sum(true_positives.flatten()) + np.sum(false_positives.flatten()))
        recall_denominator = (np.sum(true_positives.flatten()) + np.sum(false_negatives.flatten()))
        if prec_denominator > 0 and recall_denominator > 0:
            mPrecision = np.sum(true_positives.flatten()) / prec_denominator
            mRecall = np.sum(true_positives.flatten()) / recall_denominator
        else:
            return 0.0
        mF1 = 0
        if mPrecision > 0 and mRecall > 0:
            mF1 = 2 * mPrecision * mRecall / (mPrecision + mRecall)
        if reset:
            self.predictions = np.empty((0, 4), dtype=float)
            self.ground_truth = np.empty((0), dtype=int)
        return mF1