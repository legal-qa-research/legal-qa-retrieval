from typing import List

import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor
from tqdm import tqdm

from data_processor.entities.article_identity import ArticleIdentity
from utils.utilities import get_raw_from_preproc, predict_relevance_article, write_submission, build_private_data


class InferProcess:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self):
        assert self.args.load_chk_point is not None, 'Must specify the checkpoint path'
        return SentenceTransformer(model_name_or_path=self.args.load_chk_point, device=self.device).eval()

    def start_inference(self, is_choose_threshold=False):
        model = self.load_model()
        ques_pool, arti_pool, cached_rel = build_private_data()
        lis_encoded_question: List[Tensor] = model.encode(
            sentences=[get_raw_from_preproc(ques) for ques in ques_pool.proc_ques_pool])
        lis_pred_article_threshold: List[List[ArticleIdentity]] = []
        lis_pred_article_top_k: List[List[ArticleIdentity]] = []
        for i, encoded_question in enumerate(tqdm(lis_encoded_question)):
            threshold_result, top_k_result = predict_relevance_article(encoded_ques=encoded_question, model=model,
                                                                       top_n_aid=cached_rel[i],
                                                                       arti_pool=arti_pool,
                                                                       threshold=self.args.threshold,
                                                                       top_k=self.args.infer_top_k)
            lis_pred_article_threshold.append(threshold_result)
            lis_pred_article_top_k.append(top_k_result)

        for i in range(len(ques_pool.lis_ques)):
            ques_pool.lis_ques[i].relevance_articles = lis_pred_article_threshold[i]
        write_submission(lis_ques=ques_pool.lis_ques, fn='threshold_infer.json')

        for i in range(len(ques_pool.lis_ques)):
            ques_pool.lis_ques[i].relevance_articles = lis_pred_article_top_k[i]
        write_submission(lis_ques=ques_pool.lis_ques, fn='top_k_infer.json')
