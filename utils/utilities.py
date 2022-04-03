import json
import pickle
from random import shuffle

from typing import List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from torch import Tensor

from data_processor.article_pool import ArticlePool
from data_processor.entities.article_identity import ArticleIdentity
from data_processor.entities.question import Question
from data_processor.question_pool import QuestionPool
from utils.constant import pkl_private_question_pool, pkl_article_pool, pkl_private_cached_rel


def build_private_data():
    ques_pool: QuestionPool = pickle.load(open(pkl_private_question_pool, 'rb'))
    arti_pool: ArticlePool = pickle.load(open(pkl_article_pool, 'rb'))
    cached_rel = pickle.load(open(pkl_private_cached_rel, 'rb'))
    return ques_pool, arti_pool, cached_rel


def get_raw_from_preproc(preproc):
    return ' '.join([' '.join(sent) for sent in preproc])


def get_flat_list_from_preproc(preproc: List[List[str]]) -> List[str]:
    return [tok for sent in preproc for tok in sent]


def split_ids(n_samples: int, test_size=0.2):
    train_size = 1 - test_size
    cut_pos = int(n_samples * train_size)

    lis_id = [i for i in range(n_samples)]
    shuffle(lis_id)
    return lis_id[:cut_pos], lis_id[cut_pos:]


def predict_relevance_article(model: SentenceTransformer,
                              encoded_ques: Tensor,
                              top_n_aid: List[int],
                              arti_pool: ArticlePool,
                              threshold: float, top_k: int
                              ) -> Tuple[List[ArticleIdentity], List[ArticleIdentity]]:
    lis_raw_article = [get_raw_from_preproc(arti_pool.proc_text_pool[aid]) for aid in top_n_aid]
    lis_encoded_article = model.encode(sentences=lis_raw_article)
    cosim_matrix = util.cos_sim(torch.Tensor(np.array([encoded_ques])), lis_encoded_article)
    aid_sorted = np.argsort(cosim_matrix[0])[-top_k:]
    lis_aid_top_k = [arti_pool.article_identity[top_n_aid[i]] for i in aid_sorted]
    lis_aid_threshold = [arti_pool.article_identity[top_n_aid[i]]
                         for i, is_greater in enumerate(cosim_matrix[0] >= threshold) if is_greater]
    return lis_aid_threshold, lis_aid_top_k


def is_contained(lis_aid: List[ArticleIdentity], aid: ArticleIdentity):
    for c_aid in lis_aid:
        if c_aid == aid:
            return True
    return False


esp = 1e-10


def calculate_single_f2score(predict_aid: List[ArticleIdentity], true_aid: List[ArticleIdentity]) -> float:
    n_true_predict = len([aid for aid in predict_aid if is_contained(true_aid, aid)])
    precision = n_true_predict / (len(predict_aid) + esp)
    recall = n_true_predict / (len(true_aid) + esp)
    return (5 * precision * recall) / (4 * precision + recall + esp)


def calculate_f2score(predict_aid: List[List[ArticleIdentity]], true_aid: List[List[ArticleIdentity]]) -> float:
    n_ques = len(predict_aid)
    total_f2i = 0
    for i in range(n_ques):
        f2i = calculate_single_f2score(predict_aid[i], true_aid[i])
        total_f2i += f2i
    return total_f2i / (n_ques + esp)


def write_submission(lis_ques: List[Question], fn: str):
    result = []
    for ques in lis_ques:
        result.append(
            {
                'question_id': ques.question_id,
                'relevant_articles': [
                    {'law_id': aid.law_id, 'article_id': aid.article_id} for aid in ques.relevance_articles
                ]
            }
        )
    json.dump(result, open(fn, 'w'))
