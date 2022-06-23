import pickle
from typing import List

from rank_bm25 import BM25Okapi

from bm25_ranking.bm25_ranker import Bm25Ranker
from data_processor.article_pool import ArticlePool
from data_processor.question_pool import QuestionPool
from utils.constant import pkl_article_pool, pkl_question_pool, pkl_bm25okapi, pkl_bm25_infer_result
from utils.infer_result import ArticleRelevantScore, InferResult
from utils.utilities import build_public_test_data, get_flat_list_from_preproc, build_private_data


class Bm25InferTest:
    def __init__(self):
        self.qp, self.ap, self.cached_rel = build_private_data()
        self.bm25okapi: BM25Okapi = pickle.load(open(pkl_bm25okapi, 'rb'))

    def start_build_relevant_score(self):
        list_infer_result: List[InferResult] = []
        for i, proc_ques in enumerate(self.qp.proc_ques_pool):
            qid = self.qp.lis_ques[i].question_id
            ques_list_token = get_flat_list_from_preproc(self.qp.proc_ques_pool[i])
            list_score = self.bm25okapi.get_scores(ques_list_token)
            list_article_score: List[ArticleRelevantScore] = []
            for aidx in self.cached_rel[i]:
                article_identity = self.ap.article_identity[aidx]
                relevant_score = list_score[aidx]
                list_article_score.append(ArticleRelevantScore(article_identity, relevant_score))
            list_infer_result.append(InferResult(qid, list_article_score))

        pickle.dump(list_infer_result, open(pkl_bm25_infer_result, 'wb'))


if __name__ == '__main__':
    bm25_infer_test = Bm25InferTest()
    bm25_infer_test.start_build_relevant_score()
