import logging
import os
import csv

from sentence_transformers import InputExample
from sentence_transformers.evaluation import SentenceEvaluator
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from sklearn.metrics import average_precision_score
import numpy as np
from typing import List

from utils.utilities import calculate_percent_diff

logger = logging.getLogger(__name__)

esp = 1e-10


class RetrievalEvaluatorF2(SentenceEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the accuracy of identifying similar and
    dissimilar sentences.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the accuracy with a specified metric.

    The results are written in a CSV. If a CSV already exists, then values are appended.

    The labels need to be 0 for dissimilar pairs and 1 for similar pairs.

    :param sentences1: The first column of sentences - according to question
    :param sentences2: The second column of sentences - according to legal article
    :param labels: labels[i] is the label for the pair (sentences1[i], sentences2[i]). Must be 0 or 1
    :param name: Name for the output
    :param batch_size: Batch size used to compute embeddings
    :param show_progress_bar: If true, prints a progress bar
    :param write_csv: Write results to a CSV file
    """

    def __init__(self, sentences1: List[str], sentences2: List[str], labels: List[int], name: str = '',
                 batch_size: int = 32, show_progress_bar: bool = False, write_csv: bool = True):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.labels)
        for label in labels:
            assert (label == 0 or label == 1)

        self.write_csv = write_csv
        self.name = name
        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (
                    logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        # 'f2_threshold': f2_threshold,
        # 'f2_best_threshold': best_f2_threshold,
        # 'f2_top_k': f2_top_k,
        # 'f2_best_top_k': best_f2_top_k
        self.csv_file = "f2score_evaluation" + ("_" + name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps",
                            "cossim_accuracy", "cossim_accuracy_threshold", "cossim_f1", "cossim_precision",
                            "cossim_recall", "cossim_f1_threshold", "cossim_ap",
                            "cossim_f2_threshold", "cossim_f2_best_threshold",
                            "cossim_f2_top_k", "cossim_f2_best_top_k",
                            "cossim_f2_trail_threshold", "cossim_f2_best_trail_threshold",
                            "manhatten_accuracy", "manhatten_accuracy_threshold", "manhatten_f1", "manhatten_precision",
                            "manhatten_recall", "manhatten_f1_threshold", "manhatten_ap",
                            "euclidean_accuracy", "euclidean_accuracy_threshold", "euclidean_f1", "euclidean_precision",
                            "euclidean_recall", "euclidean_f1_threshold", "euclidean_ap",
                            "dot_accuracy", "dot_accuracy_threshold", "dot_f1", "dot_precision", "dot_recall",
                            "dot_f1_threshold", "dot_ap"]

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentences1 = []
        sentences2 = []
        scores = []

        for example in examples:
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
            scores.append(example.label)
        return cls(sentences1, sentences2, scores, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logger.info("Binary Accuracy Evaluation of the model on " + self.name + " dataset" + out_txt)

        scores = self.compute_metrics(model)

        # Main score is the max of Average Precision (AP)
        # main_score = max(scores[short_name]['ap'] for short_name in scores)
        main_score = max(scores['cossim']['f2_threshold'], scores['cossim']['f2_top_k'])

        file_output_data = [epoch, steps]

        for header_name in self.csv_headers:
            if '_' in header_name:
                sim_fct, metric = header_name.split("_", maxsplit=1)
                file_output_data.append(scores[sim_fct][metric])

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline='', mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow(file_output_data)
            else:
                with open(csv_path, newline='', mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(file_output_data)

        return main_score

    def compute_metrics(self, model):
        sentences = list(set(self.sentences1 + self.sentences2))
        embeddings = model.encode(sentences, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar,
                                  convert_to_numpy=True)
        emb_dict = {sent: emb for sent, emb in zip(sentences, embeddings)}
        embeddings1 = [emb_dict[sent] for sent in self.sentences1]
        embeddings2 = [emb_dict[sent] for sent in self.sentences2]

        cosine_scores = 1 - paired_cosine_distances(embeddings1, embeddings2)
        manhattan_distances = paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = paired_euclidean_distances(embeddings1, embeddings2)

        embeddings1_np = np.asarray(embeddings1)
        embeddings2_np = np.asarray(embeddings2)
        dot_scores = [np.dot(embeddings1_np[i], embeddings2_np[i]) for i in range(len(embeddings1_np))]

        labels = np.asarray(self.labels)
        output_scores = {}
        for short_name, name, scores, reverse in [['cossim', 'Cosine-Similarity', cosine_scores, True],
                                                  ['manhatten', 'Manhatten-Distance', manhattan_distances, False],
                                                  ['euclidean', 'Euclidean-Distance', euclidean_distances, False],
                                                  ['dot', 'Dot-Product', dot_scores, True]]:
            acc, acc_threshold = self.find_best_acc_and_threshold(scores, labels, reverse)
            (f2_threshold, best_f2_threshold,
             f2_top_k, best_f2_top_k,
             f2_trail_threshold, best_f2_trail_threshold) = self.find_best_f2score_and_threshold(
                sentences1=self.sentences1,
                scores=cosine_scores,
                label=self.labels)
            f1, precision, recall, f1_threshold = self.find_best_f1_and_threshold(scores, labels, reverse)
            ap = average_precision_score(labels, scores * (1 if reverse else -1))

            logger.info(
                "Accuracy with {}:           {:.2f}\t(Threshold: {:.4f})".format(name, acc * 100, acc_threshold))
            logger.info("F1 with {}:                 {:.2f}\t(Threshold: {:.4f})".format(name, f1 * 100, f1_threshold))
            logger.info("Precision with {}:          {:.2f}".format(name, precision * 100))
            logger.info("Recall with {}:             {:.2f}".format(name, recall * 100))
            logger.info("Average Precision with {}:  {:.2f}\n".format(name, ap * 100))
            logger.info(f'Choose by threshold | F2score: {f2_threshold} with threshold: {best_f2_threshold}')
            logger.info(f'Choose by top_k | F2score: {f2_top_k} with threshold: {best_f2_top_k}')

            output_scores[short_name] = {
                'accuracy': acc,
                'accuracy_threshold': acc_threshold,
                'f1': f1,
                'f1_threshold': f1_threshold,
                'precision': precision,
                'recall': recall,
                'ap': ap,
                'f2_threshold': f2_threshold,
                'f2_best_threshold': best_f2_threshold,
                'f2_top_k': f2_top_k,
                'f2_best_top_k': best_f2_top_k,
                'f2_trail_threshold': f2_trail_threshold,
                'f2_best_trail_threshold': best_f2_trail_threshold
            }

        return output_scores

    @staticmethod
    def calculate_f2score(pred_label, true_label):
        true_pos = len([i for i in range(len(pred_label)) if true_label[i] == 1 and pred_label[i] == 1])
        precision = true_pos / (len([i for i in pred_label if i == 1]) + esp)
        recall = true_pos / (len([i for i in true_label if i == 1]) + esp)
        return (5 * precision * recall) / (4 * precision + recall + esp)

    @staticmethod
    def find_best_f2score_and_threshold(sentences1: List[str], scores: np.ndarray, label: List[int]):
        # Tao set cac cau hoi co trong tap dev
        ques_pool = set(sentences1)
        # Khoi tao dict luu do lien quan cua cau hoi voi cac dieu luat
        scores_dict = {
            sentences: [] for sentences in ques_pool
        }
        # Khoi tao dict luu nhan du doan cua cau hoi voi cac dieu luat
        label_dict = {
            sentences: [] for sentences in ques_pool
        }
        # Fill score vao cac dict vua khoi tao
        for i in range(len(sentences1)):
            scores_dict[sentences1[i]].append(scores[i])
            label_dict[sentences1[i]].append(label[i])

        # Sort chi so cua dieu luat theo do lien quan voi cau hoi
        arg_sort_scores_dict = {
            sentences: np.argsort(scores_dict[sentences]) for sentences in ques_pool
        }

        # Chon threshold tot nhat cho tap dev
        max_f2score_threshold = -1
        best_threshold = None
        for threshold in np.arange(-1, 1, 0.01):
            total_f2 = 0

            for ques in ques_pool:
                ques_scores = scores_dict[ques]
                ques_label = label_dict[ques]
                ques_pred_label = [int(s >= threshold) for s in ques_scores]
                total_f2 += RetrievalEvaluatorF2.calculate_f2score(pred_label=ques_pred_label, true_label=ques_label)

            avg_f2 = total_f2 / (len(ques_pool) + esp)
            if avg_f2 > max_f2score_threshold:
                max_f2score_threshold = avg_f2
                best_threshold = threshold

        # Chon top-k tot nhat cho tap dev
        max_f2score_top_k = -1
        best_top_k = None
        for top_k in range(10):
            total_f2 = 0
            for ques in ques_pool:
                ques_label = label_dict[ques]
                ques_scores = scores_dict[ques]
                ques_pred_label = np.zeros(shape=(len(ques_scores),), dtype=int)
                for idx in arg_sort_scores_dict[ques][-top_k:]:
                    ques_pred_label[idx] = 1
                total_f2 += RetrievalEvaluatorF2.calculate_f2score(pred_label=ques_pred_label, true_label=ques_label)

            avg_f2 = total_f2 / (len(ques_pool) + esp)
            if avg_f2 > max_f2score_top_k:
                max_f2score_top_k = avg_f2
                best_top_k = top_k

        # Chon trail_threshold tot nhat cho tap dev
        max_f2score_trail_threshold = -1
        best_trail_threshold = None
        for trail_threshold in np.arange(0, 1, 0.01):
            total_f2 = 0
            for ques in ques_pool:
                ques_labels = label_dict[ques]
                ques_scores = scores_dict[ques]
                highest_score = arg_sort_scores_dict[ques][-1]
                ques_pred_label = [int(calculate_percent_diff(highest_score, s) <= trail_threshold)
                                   for s in ques_scores]
                total_f2 += RetrievalEvaluatorF2.calculate_f2score(pred_label=ques_pred_label, true_label=ques_labels)

            avg_f2 = total_f2 / (len(ques_pool) + esp)
            if avg_f2 > max_f2score_trail_threshold:
                max_f2score_trail_threshold = avg_f2
                best_trail_threshold = trail_threshold

        return (max_f2score_threshold, best_threshold,
                max_f2score_top_k, best_top_k,
                max_f2score_trail_threshold, best_trail_threshold)

    @staticmethod
    def find_best_acc_and_threshold(scores, labels, high_score_more_similar: bool):
        assert len(scores) == len(labels)
        rows = list(zip(scores, labels))

        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        max_acc = 0
        best_threshold = -1

        positive_so_far = 0
        remaining_negatives = sum(labels == 0)

        for i in range(len(rows) - 1):
            score, label = rows[i]
            if label == 1:
                positive_so_far += 1
            else:
                remaining_negatives -= 1

            acc = (positive_so_far + remaining_negatives) / len(labels)
            if acc > max_acc:
                max_acc = acc
                best_threshold = (rows[i][0] + rows[i + 1][0]) / 2

        return max_acc, best_threshold

    @staticmethod
    def find_best_f1_and_threshold(scores, labels, high_score_more_similar: bool):
        assert len(scores) == len(labels)

        scores = np.asarray(scores)
        labels = np.asarray(labels)

        rows = list(zip(scores, labels))

        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        best_f1 = best_precision = best_recall = 0
        threshold = 0
        nextract = 0
        ncorrect = 0
        total_num_duplicates = sum(labels)

        for i in range(len(rows) - 1):
            score, label = rows[i]
            nextract += 1

            if label == 1:
                ncorrect += 1

            if ncorrect > 0:
                precision = ncorrect / nextract
                recall = ncorrect / total_num_duplicates
                f1 = 2 * precision * recall / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_precision = precision
                    best_recall = recall
                    threshold = (rows[i][0] + rows[i + 1][0]) / 2

        return best_f1, best_precision, best_recall, threshold
