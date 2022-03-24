import pickle

import torch
from sentence_transformers import SentenceTransformer

from bm25_ranking.bm25_ranker import Bm25Ranker
from data_processor.article_pool import ArticlePool
from data_processor.question_pool import QuestionPool


class InferProcess:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def load_model(self):
        assert self.args.chkpoint is not None, 'Must specify the checkpoint path'
        return SentenceTransformer(model_name_or_path=self.args.chkpoint).eval()

    def build_data(self):
        assert self.args.test_file is not None, 'Must specify the test file path'
        print('Building Bm25Ranker ... ')
        ques_pool: QuestionPool = QuestionPool(ques_json_path=self.args.test_file)
        arti_pool: ArticlePool = pickle.load(open('pkl_file/article_pool.pkl', 'rb'))
        bm25ranker: Bm25Ranker = Bm25Ranker(ques_pool=ques_pool, article_pool=arti_pool)
        print('Build Bm25Ranker done !')
        return ques_pool, arti_pool, bm25ranker

    def start_inference(self):
        model = self.load_model()
        ques_pool, arti_pool, bm25ranker = self.build_data()

