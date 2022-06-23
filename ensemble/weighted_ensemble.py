import pickle
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from data_processor.article_pool import ArticlePool
from data_processor.entities.article_identity import ArticleIdentity
from data_processor.question_pool import QuestionPool
from utils.constant import pkl_sent_bert_v4_infer_result, pkl_sent_bert_v7_infer_result, pkl_bm25_infer_result, \
    pkl_xgboost_infer_result, ensemble_log
from sklearn.preprocessing import MinMaxScaler

from utils.evaluating_submission import ESP
from utils.infer_result import InferResult, ArticleRelevantScore
from utils.utilities import build_public_test_data, calculate_percent_diff, calculate_f2score


def normalize_score(normalize_list: List[InferResult]):
    list_all_score = [[infer_result.relevant_score] for ques_infer in normalize_list
                      for infer_result in ques_infer.list_infer]
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(list_all_score)
    for ques_infer in normalize_list:
        unscaled_score = [[infer_result.relevant_score] for infer_result in ques_infer.list_infer]
        scaled_score = min_max_scaler.transform(unscaled_score)
        for i in range(len(ques_infer.list_infer)):
            ques_infer.list_infer[i].relevant_score = scaled_score[i][0]


def combine_float_score(list_sw: List[Tuple[float, float]]) -> float:
    total = 0
    sum_weight = 0
    for weight, score in list_sw:
        total += weight * score
        sum_weight += weight
    return total / (sum_weight + ESP)


class WeightedEnsemble:
    def __init__(self):
        self.ques_pool, self.arti_pool, self.cached_rel = build_public_test_data()
        self.model_v4: List[InferResult] = pickle.load(open(pkl_sent_bert_v4_infer_result, 'rb'))
        self.model_v7: List[InferResult] = pickle.load(open(pkl_sent_bert_v7_infer_result, 'rb'))
        self.bm25: List[InferResult] = pickle.load(open(pkl_bm25_infer_result, 'rb'))
        self.xgboost: List[InferResult] = pickle.load(open(pkl_xgboost_infer_result, 'rb'))
        self.trail_threshold = 0.03
        assert self.check_data_synchronous(), 'Data is not synchronous'
        print('Init class successfully')

    def check_data_synchronous(self):
        for i in range(len(self.bm25)):
            qid = self.bm25[i].qid
            if self.model_v4[i].qid != qid or self.model_v7[i].qid != qid or self.xgboost[i].qid:
                return False
            for j in range(len(self.bm25[i].list_infer)):
                aid = self.bm25[i].list_infer[j].article_identity
                if self.model_v4[i].list_infer[j].article_identity != aid:
                    return False
                if self.model_v7[i].list_infer[j].article_identity != aid:
                    return False
                if self.xgboost[i].list_infer[j].article_identity != aid:
                    return False
        return True

    def combine_model_score(self, alpha_model_v4: float, alpha_model_v7: float,
                            alpha_xgboost: float, alpha_bm25: float) -> List[InferResult]:
        list_infer_result: List[InferResult] = []
        # Ket hop diem theo trong so
        for i in range(len(self.bm25)):
            qid = self.bm25[i].qid
            list_article_score: List[ArticleRelevantScore] = []
            for j in range(len(self.bm25[i].list_infer)):
                aid = self.bm25[i].list_infer[j].article_identity
                combined_score = combine_float_score([
                    (alpha_model_v4, self.model_v4[i].list_infer[j].relevant_score),
                    (alpha_model_v7, self.model_v7[i].list_infer[j].relevant_score),
                    (alpha_bm25, self.bm25[i].list_infer[j].relevant_score),
                    (alpha_xgboost, self.xgboost[i].list_infer[j].relevant_score)
                ])
                list_article_score.append(ArticleRelevantScore(aid, combined_score))
            list_infer_result.append(InferResult(qid, list_article_score))

        # Chuan hoa diem ket hop thu duoc
        normalize_score(list_infer_result)

        return list_infer_result

    def calculate_f2_score(self, lis_infer: List[InferResult]) -> float:
        # Tinh diem F2score dua vao diem lien quan voi chien luoc du doan trail_threshold
        test_predict_article: List[List[ArticleIdentity]] = []
        test_true_article: List[List[ArticleIdentity]] = []
        for i, ques_infer in enumerate(lis_infer):
            qid = ques_infer.qid
            assert qid == self.ques_pool.lis_ques[i].question_id, 'Question is not synchronous'
            test_true_article.append(self.ques_pool.lis_ques[i].relevance_articles)
            highest_relevant_aid = np.max([article_infer.relevant_score for article_infer in ques_infer.list_infer])
            list_predict: List[ArticleIdentity] = []
            for article_infer in ques_infer.list_infer:
                if calculate_percent_diff(highest_relevant_aid, article_infer.relevant_score) <= self.trail_threshold:
                    list_predict.append(article_infer.article_identity)
            test_predict_article.append(list_predict)
        return calculate_f2score(test_predict_article, test_true_article)

    def start_ensemble(self):
        # Chuan hoa diem danh gia cua moi mo hinh
        normalize_score(self.model_v4)
        normalize_score(self.bm25)
        normalize_score(self.model_v7)
        normalize_score(self.xgboost)

        # Tao file log lai qua trinh tim kiem weight
        f = open(ensemble_log, 'w')

        # Tim kiem trong so cua moi mo hinh
        max_f2_score: float = 0
        best_weight: Tuple[float, float, float] = (0, 0, 0)
        for alpha_model_v4 in tqdm(np.arange(0, 1, 0.01), desc='For weight model v4'):
            for alpha_model_v7 in np.arange(0, 1 - alpha_model_v4, 0.01):
                for alpha_xgboost in np.arange(0, 1 - alpha_model_v4 - alpha_model_v7, 0.01):
                    alpha_bm25 = 1 - alpha_model_v4 - alpha_model_v7 - alpha_xgboost
                    lis_combined_infer = self.combine_model_score(alpha_model_v4, alpha_model_v7,
                                                                  alpha_bm25, alpha_xgboost)
                    f2_score = self.calculate_f2_score(lis_combined_infer)
                    f.write(f'alpha_model_v4: {alpha_model_v4}')
                    f.write(f'alpha_model_v7: {alpha_model_v7}')
                    f.write(f'alpha_xgboost: {alpha_xgboost}')
                    f.write(f'alpha_bm25: {alpha_bm25} \n')
                    f.write(f'f2-score: {f2_score} \n')
                    if f2_score > max_f2_score:
                        max_f2_score = f2_score
                        best_weight = (alpha_model_v4, alpha_model_v7, alpha_bm25)

        # Log lai tap weight co diem f2 cao nhat
        f.write(f'best_f2_score: {max_f2_score} | best_weight: {best_weight}')
        f.close()


if __name__ == '__main__':
    weighted_ensemble = WeightedEnsemble()
    weighted_ensemble.start_ensemble()
