import pickle

import numpy as np
from rank_bm25 import BM25Okapi

from data_processor.article_pool import ArticlePool
from data_processor.question_pool import QuestionPool
from utils.utilities import get_raw_from_preproc


class Bm25Ranker:
    def __init__(self, article_pool: ArticlePool, ques_pool: QuestionPool):
        self.article_pool = article_pool
        self.ques_pool = ques_pool
        corpus = [get_raw_from_preproc(preproc_text) for preproc_text in self.article_pool.proc_text_pool]
        self.bm25okapi = BM25Okapi(corpus)
        self.__raw_ques = [get_raw_from_preproc(preproc_text) for preproc_text in self.ques_pool.proc_ques_pool]

    def get_topn(self, ques_id: int, top_n: int):
        lis_score = self.bm25okapi.get_scores(self.__raw_ques[ques_id])
        return np.argsort(lis_score)[-top_n:]


if __name__ == '__main__':
    ap: ArticlePool = pickle.load(open('data_processor/article_pool.pkl', 'rb'))
    qp: QuestionPool = pickle.load(open('data_processor/question_pool.pkl', 'rb'))
    br: Bm25Ranker = pickle.load(open('bm25_ranking/bm25_ranker.pkl', 'rb'))
    # br = Bm25Ranker(ap, qp)
    # pickle.dump(br, open('bm25_ranking/bm25_ranker.pkl', 'wb'))
    test_id = 100
    topn = 1000
    print(qp.proc_ques_pool[test_id])
    topn_article = br.get_topn(test_id, topn)
    print(topn_article)
    rela = qp.lis_ques[test_id].relevance_articles[0]
    print(ap.get_position(rela) in topn_article)
