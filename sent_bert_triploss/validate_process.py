import pickle
from typing import List

import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor
from tqdm import tqdm

from data_processor.article_pool import ArticlePool
from data_processor.entities.article_identity import ArticleIdentity
from data_processor.question_pool import QuestionPool
from sent_bert_triploss.constant import pkl_private_question_pool, pkl_article_pool, \
    pkl_private_cached_rel
from utils.utilities import get_raw_from_preproc, predict_relevance_article, write_submission


class ValidateProcess:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self):
        assert self.args.chkpoint is not None, 'Must specify the checkpoint path'
        return SentenceTransformer(model_name_or_path=self.args.chkpoint, device=self.device).eval()

    @staticmethod
    def build_data():
        ques_pool: QuestionPool = pickle.load(open(pkl_private_question_pool, 'rb'))
        arti_pool: ArticlePool = pickle.load(open(pkl_article_pool, 'rb'))
        cached_rel = pickle.load(open(pkl_private_cached_rel, 'rb'))
        return ques_pool, arti_pool, cached_rel

    def start_inference(self, is_choose_threshold=False):
        model = self.load_model()
        ques_pool, arti_pool, cached_rel = self.build_data()
        lis_encoded_question: List[Tensor] = model.encode(
            sentences=[get_raw_from_preproc(ques) for ques in ques_pool.proc_ques_pool])
        lis_pred_article: List[List[ArticleIdentity]] = []
        for i, encoded_question in enumerate(tqdm(lis_encoded_question)):
            lis_pred_article.append(
                predict_relevance_article(encoded_ques=encoded_question, model=model, top_n_aid=cached_rel[i],
                                          arti_pool=arti_pool))
        for i in range(len(ques_pool.lis_ques)):
            ques_pool.lis_ques[i].relevance_articles = lis_pred_article[i]
        write_submission(ques_pool.lis_ques)
