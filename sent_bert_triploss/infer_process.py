import pickle
from typing import List

import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor
from tqdm import tqdm

from data_processor.article_pool import ArticlePool
from data_processor.entities.article_identity import ArticleIdentity
from data_processor.question_pool import QuestionPool
from utils.constant import pkl_question_pool, pkl_article_pool, pkl_cached_rel, pkl_split_ids
from utils.utilities import get_raw_from_preproc, predict_relevance_article, write_submission, build_private_data, \
    calculate_f2score


class InferProcess:
    def __init__(self, args):
        self.args = args
        self.device = self.args if torch.cuda.is_available() else 'cpu'

    def load_model(self):
        assert self.args.load_chk_point is not None, 'Must specify the checkpoint path'
        return SentenceTransformer(model_name_or_path=self.args.load_chk_point, device=self.device).eval()

    def infer_sample(self, ques_pool: QuestionPool, arti_pool: ArticlePool, cached_rel: List[List[int]]):
        model = self.load_model()
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
        return lis_pred_article_threshold, lis_pred_article_top_k

    def start_test(self):
        test_ids = pickle.load(open(pkl_split_ids, 'rb'))['test']
        ques_pool: QuestionPool = pickle.load(open(pkl_question_pool, 'rb'))
        subset_ques_pool = ques_pool.extract_sub_set(test_ids)

        cached_rel = pickle.load(open(pkl_cached_rel, 'rb'))
        subset_cached_rel = [cached_rel[i] for i in test_ids]

        arti_pool: ArticlePool = pickle.load(open(pkl_article_pool, 'rb'))
        lis_pred_article_threshold, lis_pred_article_top_k = self.infer_sample(subset_ques_pool, arti_pool,
                                                                               subset_cached_rel)
        lis_true_article = [ques.relevance_articles for ques in subset_ques_pool.lis_ques]
        print('F2-score on threshold strategy: ', calculate_f2score(lis_pred_article_threshold, lis_true_article))
        print('F2-score on top-k strategy: ', calculate_f2score(lis_pred_article_top_k, lis_true_article))

    def start_inference(self):
        ques_pool, arti_pool, cached_rel = build_private_data()
        lis_pred_article_threshold, lis_pred_article_top_k = self.infer_sample(ques_pool, arti_pool, cached_rel)

        for i in range(len(ques_pool.lis_ques)):
            ques_pool.lis_ques[i].relevance_articles = lis_pred_article_threshold[i]
        write_submission(lis_ques=ques_pool.lis_ques, fn='threshold_infer.json')

        for i in range(len(ques_pool.lis_ques)):
            ques_pool.lis_ques[i].relevance_articles = lis_pred_article_top_k[i]
        write_submission(lis_ques=ques_pool.lis_ques, fn='top_k_infer.json')
