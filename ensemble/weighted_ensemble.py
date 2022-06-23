import pickle
from typing import List

from utils.constant import pkl_sent_bert_v4_infer_result, pkl_sent_bert_v7_infer_result, pkl_bm25_infer_result, \
    pkl_xgboost_infer_result
from utils.infer_result import InferResult


class WeightedEnsemble:
    def __init__(self):
        self.model_v4: List[InferResult] = pickle.load(open(pkl_sent_bert_v4_infer_result, 'rb'))
        self.model_v7: List[InferResult] = pickle.load(open(pkl_sent_bert_v7_infer_result, 'rb'))
        self.bm25: List[InferResult] = pickle.load(open(pkl_bm25_infer_result, 'rb'))
        self.xgboost: List[InferResult] = pickle.load(open(pkl_xgboost_infer_result, 'rb'))

    def start_ensemble(self):
        pass


if __name__ == '__main__':
    weighted_ensemble = WeightedEnsemble()
    weighted_ensemble.start_ensemble()
