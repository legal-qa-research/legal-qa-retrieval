import pickle
from typing import List, Tuple

import fasttext
import numpy as np
from tqdm import tqdm

from data_processor.article_pool import ArticlePool
from data_processor.question_pool import QuestionPool
from traditional_ml.args_management import args
from traditional_ml.constant import fasttext_model
from traditional_ml.raw_input_example import RawInputExample
from sklearn import svm
from utils.utilities import build_private_data, get_flat_list_from_preproc, build_vec, write_submission


class RunInferProcess:
    def __init__(self, args):
        self.ques_pool, self.arti_pool, self.cached_rel = build_private_data()
        self.fasttext_model = fasttext.load_model(fasttext_model)
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

    def predict_threshold(self, lis_raw_inp_exp: List[RawInputExample]):
        self.reset_ques_pool()

        for i in range(len(lis_raw_inp_exp)):
            aid = lis_raw_inp_exp[i].article_id
            qid = lis_raw_inp_exp[i].ques_id
            prob = lis_raw_inp_exp[i].prob
            if prob >= self.args.infer_threshold:
                self.ques_pool.lis_ques[qid].relevance_articles.append(self.arti_pool.article_identity[aid])
        write_submission(lis_ques=self.ques_pool.lis_ques, fn='trad_ml_svm_threshold.json')

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

        write_submission(self.ques_pool.lis_ques, fn='trad_ml_svm_threshold.json')

    def start_infer(self):
        lis_raw_inp_exp = self.start_build_private_data_feature()
        model: svm.SVC = pickle.load(open('traditional_ml/model_pool/svm_model.pkl', 'rb'))

        lis_features: List[np.ndarray] = []
        for raw_ques in tqdm(lis_raw_inp_exp, desc='Building Features'):
            features = np.concatenate((build_vec(ft=self.fasttext_model, lis_tok=raw_ques.ques),
                                       build_vec(ft=self.fasttext_model, lis_tok=raw_ques.articles)), axis=0)
            lis_features.append(features)
        print('Predicting ... ')
        for i, p in enumerate(model.predict_proba(lis_features)[:, 1]):
            lis_raw_inp_exp[i].prob = p
        print('Predict done, start record result')

        self.predict_threshold(lis_raw_inp_exp)
        self.predict_top_k(lis_raw_inp_exp)


if __name__ == '__main__':
    ifp = RunInferProcess(args)
    ifp.start_infer()
