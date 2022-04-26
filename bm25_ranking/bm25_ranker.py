import pickle

import numpy as np
from rank_bm25 import BM25Okapi

from data_processor.article_pool import ArticlePool
from data_processor.question_pool import QuestionPool
from utils.utilities import get_raw_from_preproc


class Bm25Ranker:
    def __init__(self, article_pool: ArticlePool, ques_pool: QuestionPool, bm25okapi_pkl: str = None):
        self.article_pool = article_pool
        self.ques_pool = ques_pool
        self.corpus = [get_raw_from_preproc(preproc_text).split(' ') for preproc_text in
                       self.article_pool.proc_text_pool]
        if bm25okapi_pkl is None:
            print('bm25okapi is building...')
            self.bm25okapi = BM25Okapi(self.corpus)
            print('Done')
        else:
            self.bm25okapi: BM25Okapi = pickle.load(open(bm25okapi_pkl, 'rb'))
        self.raw_ques = [get_raw_from_preproc(preproc_text) for preproc_text in self.ques_pool.proc_ques_pool]

    def save_bm25okapi(self, output_path: str):
        pickle.dump(self.bm25okapi, open(output_path, 'wb'))

    def get_topn(self, ques_id: int, top_n: int):
        scores = self.bm25okapi.get_scores(query=self.raw_ques[ques_id].split(' '))
        return np.argsort(scores)[::-1][:top_n]


def build_bm25_and_save():
    ap: ArticlePool = pickle.load(open('pkl_file/kse_article_pool.pkl', 'rb'))
    qp: QuestionPool = pickle.load(open('pkl_file/kse_question_pool.pkl', 'rb'))
    br = Bm25Ranker(ap, qp)
    # br.save_bm25okapi(output_path='pkl_file/kse_bm25okapi_v1.pkl')


if __name__ == '__main__':
    build_bm25_and_save()
    # test_pkl_loader()
