import pickle
from typing import List

import fasttext
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from data_processor.article_pool import ArticlePool
from data_processor.question_pool import QuestionPool
from traditional_ml.args_management import args
from traditional_ml.constant import fasttext_model
from traditional_ml.features_builder import FeaturesBuilder
from traditional_ml.raw_input_example import RawInputExample

from utils.constant import pkl_tfidf, pkl_xgb_model, xgb_model
from utils.infer_result import ArticleRelevantScore, InferResult
from utils.utilities import build_private_data, get_flat_list_from_preproc, write_submission
from xgboost import XGBClassifier


class RunInferProcess:
    def __init__(self, args):
        self.ques_pool, self.arti_pool, self.cached_rel = build_private_data()
        self.fasttext_model = fasttext.load_model(fasttext_model)
        self.tfidf_vtr: TfidfVectorizer = pickle.load(open(pkl_tfidf, 'rb'))
        self.args = args

    def start_build_private_data_feature(self) -> List[RawInputExample]:
        ques_features: List[RawInputExample] = []
        for qid, proc_ques in enumerate(self.ques_pool.proc_ques_pool):
            flatten_ques = get_flat_list_from_preproc(proc_ques)
            for aid in self.cached_rel[qid]:
                proc_art = self.arti_pool.proc_text_pool[aid]
                flatten_article = get_flat_list_from_preproc(proc_art)
                ques_features.append(
                    RawInputExample(ques_id=qid, ques=flatten_ques, article_id=aid, articles=flatten_article, label=0))
        return ques_features

    def reset_ques_pool(self):
        for ques in self.ques_pool.lis_ques:
            ques.relevance_articles = []

    def save_predict_prob(self, lis_raw_inp_exp: List[RawInputExample]):
        infer_result_dict = {}
        for raw_inp_exp in lis_raw_inp_exp:
            aid = raw_inp_exp.article_id
            qid = raw_inp_exp.ques_id
            prob = raw_inp_exp.prob
            ars = ArticleRelevantScore(self.arti_pool.article_identity[aid], prob)
            if qid not in infer_result_dict.keys():
                infer_result_dict[qid] = [ars]
            else:
                infer_result_dict[qid].append(ars)
        test_infer_result: List[InferResult] = []
        for qid in infer_result_dict.keys():
            test_infer_result.append(InferResult(str(qid), infer_result_dict[qid]))
        pickle.dump(test_infer_result, open(f'alqac_2022_fast_text.pkl', 'wb'))

    def predict_threshold(self, lis_raw_inp_exp: List[RawInputExample]):
        self.reset_ques_pool()

        for i in range(len(lis_raw_inp_exp)):
            aid = lis_raw_inp_exp[i].article_id
            qid = lis_raw_inp_exp[i].ques_id
            prob = lis_raw_inp_exp[i].prob
            if prob >= self.args.infer_threshold:
                self.ques_pool.lis_ques[qid].relevance_articles.append(self.arti_pool.article_identity[aid])
        write_submission(lis_ques=self.ques_pool.lis_ques, fn='trad_ml_xgboost_threshold.json')

    def predict_top_k(self, lis_raw_inp_exp: List[RawInputExample]):
        self.reset_ques_pool()

        def sort_fn(x: RawInputExample):
            return x.prob

        sorted_lis = sorted(lis_raw_inp_exp, key=sort_fn)

        for raw_inp in sorted_lis:
            qid = raw_inp.ques_id
            aid = raw_inp.article_id
            if len(self.ques_pool.lis_ques[qid].relevance_articles) == self.args.infer_top_k:
                self.ques_pool.lis_ques[qid].relevance_articles.pop(0)
            self.ques_pool.lis_ques[qid].relevance_articles.append(self.arti_pool.article_identity[aid])

        write_submission(self.ques_pool.lis_ques, fn='trad_ml_xgboost_threshold.json')

    def start_infer(self):
        lis_raw_inp_exp = self.start_build_private_data_feature()
        # model: XGBClassifier = pickle.load(open(pkl_xgb_model, 'rb'))
        model: XGBClassifier = XGBClassifier()
        model.load_model(xgb_model)

        lis_features: List[np.ndarray] = []
        for raw_ques in tqdm(lis_raw_inp_exp, desc='Building Features'):
            lis_features.append(FeaturesBuilder.cal_feature_from_inp(input_exp=raw_ques, ft=self.fasttext_model,
                                                                     tfidf_vtr=self.tfidf_vtr))
        print('Predicting ... ')
        for i, p in enumerate(model.predict_proba(lis_features)[:, 1]):
            lis_raw_inp_exp[i].prob = p
        print('Predict done, start record result')

        self.predict_threshold(lis_raw_inp_exp)
        self.predict_top_k(lis_raw_inp_exp)


if __name__ == '__main__':
    ifp = RunInferProcess(args)
    ifp.start_infer()
