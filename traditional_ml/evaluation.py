import os.path
import pickle
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from data_processor.article_pool import ArticlePool
from data_processor.entities.article_identity import ArticleIdentity
from traditional_ml.constant import test_examples_path, evaluation_output
from traditional_ml.raw_input_example import RawInputExample
from utils.constant import pkl_article_pool
from utils.utilities import calculate_f2score


class Evaluation:
    def __init__(self):
        self.test_examples: List[RawInputExample] = pickle.load(open(test_examples_path, 'rb'))
        self.article_pool: ArticlePool = pickle.load(open(pkl_article_pool, 'rb'))

    def choose_best_threshold(self, eval_dict: Dict[int, List[Tuple[int, float, float]]]):
        lis_ques = eval_dict.keys()
        max_f2_score = -1
        best_threshold = None
        for threshold in np.arange(0, 1, 0.01):
            test_pred_aid: List[List[ArticleIdentity]] = []
            test_true_aid: List[List[ArticleIdentity]] = []

            for qid in lis_ques:
                pred_aid: List[ArticleIdentity] = []
                true_aid: List[ArticleIdentity] = []
                for (aid, prob, label) in eval_dict[qid]:
                    if prob >= threshold:
                        pred_aid.append(self.article_pool.article_identity[aid])
                    if label == 1:
                        true_aid.append(self.article_pool.article_identity[aid])
                test_pred_aid.append(pred_aid)
                test_true_aid.append(true_aid)

            f2score = calculate_f2score(predict_aid=test_pred_aid, true_aid=test_true_aid)
            if f2score > max_f2_score:
                max_f2_score = f2score
                best_threshold = threshold
        return max_f2_score, best_threshold

    def choose_best_top_k(self, eval_dict: Dict[int, List[Tuple[int, float, float]]]):
        lis_ques = eval_dict.keys()
        max_f2score = -1
        best_top_k = None
        for top_k in range(10):
            test_pred_aid: List[List[ArticleIdentity]] = []
            test_true_aid: List[List[ArticleIdentity]] = []

            for qid in lis_ques:
                lis_prob = [prob for (aid, prob, label) in eval_dict[qid]]
                arg_sorted_prob = np.argsort(lis_prob)[-top_k:]
                pred_aid: List[int] = [eval_dict[qid][i][0] for i in arg_sorted_prob]
                pred_aid: List[ArticleIdentity] = [self.article_pool.article_identity[aid] for aid in pred_aid]

                true_aid: List[ArticleIdentity] = []
                for (aid, prob, label) in eval_dict[qid]:
                    if label == 1:
                        true_aid.append(self.article_pool.article_identity[aid])

                test_pred_aid.append(pred_aid)
                test_true_aid.append(true_aid)
            f2score = calculate_f2score(predict_aid=test_pred_aid, true_aid=test_true_aid)
            if f2score > max_f2score:
                max_f2score = f2score
                best_top_k = top_k
        return max_f2score, best_top_k

    @staticmethod
    def write_csv(epoch, step, threshold_f2score, threshold, top_k_f2score, top_k):
        s1 = pd.DataFrame({
            'epoch': [epoch],
            'step': [step],
            'threshold_f2score': [threshold_f2score],
            'threshold': [threshold],
            'top_k_f2score': [top_k_f2score],
            'top_k': [top_k]
        })
        if not os.path.exists(evaluation_output):
            s1.to_csv(evaluation_output)
        else:
            s0 = pd.read_csv(evaluation_output)
            s0 = pd.concat([s0, s1])
            s0.to_csv(evaluation_output)

    def start_eval(self, epoch: int, step: int, y_prob: np.ndarray):
        assert len(self.test_examples) == len(y_prob), 'List probability is not suitable'
        eval_dict = {}
        for i, test_example in enumerate(self.test_examples):
            ques_id = test_example.ques_id
            aid = test_example.article_id
            label = test_example.label
            if ques_id in eval_dict.keys():
                eval_dict[ques_id].append((aid, y_prob[i][1], label))
            else:
                eval_dict[ques_id] = [(aid, y_prob[i][1], label)]

        threshold_f2_score, best_threshold = self.choose_best_threshold(eval_dict)
        top_k_f2score, best_top_k = self.choose_best_top_k(eval_dict)
        self.write_csv(epoch, step, threshold_f2_score, best_threshold, top_k_f2score, best_top_k)
