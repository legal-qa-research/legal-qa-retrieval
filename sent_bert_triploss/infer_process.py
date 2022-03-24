import pickle
from typing import List

import torch
from sentence_transformers import SentenceTransformer, util
from torch import Tensor

from data_processor.article_pool import ArticlePool
from data_processor.entities.article_identity import ArticleIdentity
from data_processor.question_pool import QuestionPool
from utils.utilities import calculate_f2score, get_raw_from_preproc, predict_relevance_article


class InferProcess:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self):
        assert self.args.chkpoint is not None, 'Must specify the checkpoint path'
        return SentenceTransformer(model_name_or_path=self.args.chkpoint, device=self.device).eval()

    @staticmethod
    def build_data():
        ques_pool: QuestionPool = pickle.load(open('pkl_file/private_question_pool.pkl', 'rb'))
        arti_pool: ArticlePool = pickle.load(open('pkl_file/article_pool.pkl', 'rb'))
        cached_rel = pickle.load(open('pkl_file/private_cached_rel.pkl', 'rb'))
        return ques_pool, arti_pool, cached_rel

    def start_inference(self, is_choose_threshold=False):
        model = self.load_model()
        ques_pool, arti_pool, cached_rel = self.build_data()
        lis_encoded_question: List[Tensor] = model.encode(
            sentences=[get_raw_from_preproc(ques) for ques in ques_pool.proc_ques_pool])
        lis_pred_article: List[List[ArticleIdentity]] = []
        for i, encoded_question in enumerate(lis_encoded_question):
            lis_pred_article.append(
                predict_relevance_article(encoded_ques=encoded_question, model=model, top_n_aid=cached_rel[i],
                                          arti_pool=arti_pool))

        lis_true_article: List[List[ArticleIdentity]] = [q.relevance_articles for q in ques_pool.lis_ques]
        print('F2score = ', calculate_f2score(predict_aid=lis_pred_article, true_aid=lis_true_article))
