import pickle

from tqdm import tqdm

from bm25_ranking.bm25_ranker import Bm25Ranker
from data_processor.article_pool import ArticlePool
from data_processor.question_pool import QuestionPool
from utils.utilities import is_contained


class Bm25RankerCached:
    def __init__(self, pkl_article_pool: str, pkl_ques_pool: str, pkl_bm25okapi: str, cached_path=None):
        self.ap: ArticlePool = pickle.load(open(pkl_article_pool, 'rb'))
        self.qp: QuestionPool = pickle.load(open(pkl_ques_pool, 'rb'))
        self.br: Bm25Ranker = Bm25Ranker(article_pool=self.ap, ques_pool=self.qp, bm25okapi_pkl=pkl_bm25okapi)
        if cached_path is not None:
            self.cached_rel_aid = pickle.load(open(cached_path, 'rb'))
        else:
            self.cached_rel_aid = None

    def start_build_cache(self, cached_path: str, top_n=100):
        cached_rel_aid = []
        for qid in tqdm(range(len(self.qp.lis_ques))):
            lis_rel_aid = self.br.get_topn(ques_id=qid, top_n=top_n)
            cached_rel_aid.append(lis_rel_aid)
        pickle.dump(cached_rel_aid, open(cached_path, 'wb'))

    def test_cached_rel(self, cached_rel_path: str):
        cached_rel_arr = pickle.load(open(cached_rel_path, 'rb'))
        total_recall = 0
        for i in range(len(self.qp.lis_ques)):
            true_rel = self.qp.lis_ques[i].relevance_articles
            pred_rel = [self.ap.article_identity[aid] for aid in cached_rel_arr[i]]
            true_pred = len([aid for aid in pred_rel if is_contained(lis_aid=true_rel, aid=aid)])
            total_recall += true_pred / len(true_rel)
        print('Average recall score = ', total_recall / len(self.qp.lis_ques))

    def get_topn(self, ques_id: int, top_n=100):
        return self.cached_rel_aid[ques_id]


if __name__ == '__main__':
    # qp = QuestionPool(ques_json_path='data/private_test_question.json')
    # preprocessor = Preprocessor()
    # qp.run_preprocess(preprocessor)
    # pickle.dump(qp, open('pkl_file/private_question_pool.pkl', 'wb'))
    # private_bm_cached = Bm25RankerCached(pkl_ques_pool='pkl_file/private_question_pool.pkl')
    # private_bm_cached.start_build_cache()
    # cached_rel = pickle.load(open('pkl_file/private_cached_rel.pkl', 'rb'))
    # print(len(cached_rel))

    bm_cached = Bm25RankerCached(pkl_bm25okapi='pkl_file/alqac_2022_bm25okapi_v1.pkl',
                                 pkl_article_pool='pkl_file/alqac_2022_article_pool.pkl',
                                 pkl_ques_pool='pkl_file/alqac_2022_test_question_pool.pkl')
    bm_cached.start_build_cache(cached_path='pkl_file/alqac_2022_test_cached_rel_v1_top100.pkl', top_n=100)

    # bm_cached.test_cached_rel(cached_rel_path='pkl_file/alqac_2022_cached_rel_v1_top100.pkl')
