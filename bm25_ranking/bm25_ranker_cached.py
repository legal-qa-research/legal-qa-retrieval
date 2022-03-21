import pickle

from tqdm import tqdm

from bm25_ranking.bm25_ranker import Bm25Ranker
from data_processor.article_pool import ArticlePool
from data_processor.question_pool import QuestionPool


class Bm25RankerCached:
    def __init__(self):
        self.ap: ArticlePool = pickle.load(open('pkl_file/article_pool.pkl', 'rb'))
        self.qp: QuestionPool = pickle.load(open('pkl_file/question_pool.pkl', 'rb'))
        self.br: Bm25Ranker = pickle.load(open('pkl_file/bm25_ranker.pkl', 'rb'))

    def start_build_cache(self, top_n=100):
        cached_rel_aid = []
        for qid in tqdm(range(len(self.qp.lis_ques))):
            lis_rel_aid = self.br.get_topn(ques_id=qid, top_n=top_n)
            cached_rel_aid.append(lis_rel_aid)
        pickle.dump(cached_rel_aid, open('pkl_file/cached_rel.pkl', 'wb'))


if __name__ == '__main__':
    bm_cached = Bm25RankerCached()
    bm_cached.start_build_cache()
    # cached_rel = pickle.load(open('pkl_file/cached_rel.pkl', 'rb'))
    # print(cached_rel)
