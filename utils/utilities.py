import json
import math
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
from utils.constant import pkl_test_question_pool, pkl_article_pool, pkl_private_cached_rel, pkl_split_ids, \
    pkl_question_pool, pkl_cached_rel
from utils.evaluating_submission import ESP


def build_private_data() -> Tuple[QuestionPool, ArticlePool, List[List[int]]]:
    ques_pool: QuestionPool = pickle.load(open(pkl_test_question_pool, 'rb'))
    arti_pool: ArticlePool = pickle.load(open(pkl_article_pool, 'rb'))
    cached_rel = pickle.load(open(pkl_private_cached_rel, 'rb'))
    return ques_pool, arti_pool, cached_rel


def build_public_test_data() -> Tuple[QuestionPool, ArticlePool, List[List[int]]]:
    # Lay id cua cac question trong tap test
    test_ids = pickle.load(open(pkl_split_ids, 'rb'))['test']
    ques_pool: QuestionPool = pickle.load(open(pkl_question_pool, 'rb'))
    # Lay cac object question trong question pool theo id
    subset_ques_pool = ques_pool.extract_sub_set(test_ids)

    cached_rel = pickle.load(open(pkl_cached_rel, 'rb'))
    # Lay cac dieu luat lien quan toi question trong tap test
    subset_cached_rel = [cached_rel[i] for i in test_ids]

    arti_pool: ArticlePool = pickle.load(open(pkl_article_pool, 'rb'))
    return subset_ques_pool, arti_pool, subset_cached_rel


def get_raw_from_preproc(preproc):
    return ' '.join([' '.join(sent) for sent in preproc])


def cut_off_text(match_text: str, target_text: str, window_size: int):
    num_match_token = len(match_text.split(' '))
    num_target_token = len(target_text.split(' '))
    if num_target_token <= window_size:
        return target_text
    start_pos = target_text.find(match_text)
    assert start_pos > -1, 'Cannot find answer in article'
    expand_size = window_size - num_match_token
    assert expand_size > 0, 'The expand size is larger than max_size'
    end_pos = start_pos + num_match_token - 1
    len_tail = num_target_token - end_pos - 1
    len_head = start_pos
    expected_len_each_size = math.ceil((window_size - num_match_token) / 2)
    if len_tail < expected_len_each_size:
        lack_len = expected_len_each_size - len_tail
        len_head += lack_len
        len_tail -= lack_len
    if len_head < expected_len_each_size:
        lack_len = expected_len_each_size - len_head
        len_head -= lack_len
        len_tail += lack_len
    return target_text[start_pos - len_head: end_pos + len_tail + 1]


def get_flat_list_from_preproc(preproc: List[List[str]]) -> List[str]:
    return [tok for sent in preproc for tok in sent]


def split_ids(n_samples: int, test_size=0.2):
    train_size = 1 - test_size
    cut_pos = int(n_samples * train_size)

    lis_id = [i for i in range(n_samples)]
    shuffle(lis_id)
    return lis_id[:cut_pos], lis_id[cut_pos:]


def calculate_percent_diff(base_score: float, score: float) -> float:
    return abs(base_score - score) / (base_score + ESP)


def get_relevant_score_with_ques(model: SentenceTransformer, encoded_ques: Tensor,
                                 top_n_aid: List[int], arti_pool: ArticlePool):
    # Bieu dien cau hoi va dieu luat thanh vector
    lis_raw_article = [get_raw_from_preproc(arti_pool.proc_text_pool[aid]) for aid in top_n_aid]
    lis_encoded_article = model.encode(sentences=lis_raw_article)

    # Tinh diem tuong quan cosim giua cau hoi va top_n dieu luat
    cosim_matrix = util.cos_sim(torch.Tensor(np.array([encoded_ques])), lis_encoded_article)
    return cosim_matrix[0]


def predict_relevance_article(relevant_score: Tensor,
                              top_n_aid: List[int],
                              arti_pool: ArticlePool,
                              threshold: float, top_k: int, trail_threshold: float
                              ) -> Tuple[List[ArticleIdentity], List[ArticleIdentity], List[ArticleIdentity]]:
    # Sap xep theo diem lien quan
    idx_sorted_relevant_score = np.argsort(relevant_score)

    # Du doan lien quan theo top-k
    top_k_aid_sorted = idx_sorted_relevant_score[-top_k:]
    lis_aid_top_k = [arti_pool.article_identity[top_n_aid[i]] for i in top_k_aid_sorted]

    # Du doan lien quan theo threshold
    lis_aid_threshold = [arti_pool.article_identity[top_n_aid[i]]
                         for i, is_greater in enumerate(relevant_score >= threshold) if is_greater]

    # Du doan lien quan theo trail_threshold
    lis_aid_trail_threshold: List[ArticleIdentity] = []
    highest_relevant_score = relevant_score[idx_sorted_relevant_score[-1]]
    for idx in range(len(top_n_aid)):
        if calculate_percent_diff(highest_relevant_score, relevant_score[idx]) <= trail_threshold:
            lis_aid_trail_threshold.append(arti_pool.article_identity[top_n_aid[idx]])

    return lis_aid_threshold, lis_aid_top_k, lis_aid_trail_threshold


def predict_relevance_article_old(model: SentenceTransformer,
                                  encoded_ques: Tensor,
                                  top_n_aid: List[int],
                                  arti_pool: ArticlePool,
                                  threshold: float, top_k: int, trail_threshold: float
                                  ) -> Tuple[List[ArticleIdentity], List[ArticleIdentity], List[ArticleIdentity]]:
    # Bieu dien cau hoi va dieu luat thanh vector
    lis_raw_article = [get_raw_from_preproc(arti_pool.proc_text_pool[aid]) for aid in top_n_aid]
    lis_encoded_article = model.encode(sentences=lis_raw_article)

    # Tinh diem tuong quan cosim giua cau hoi va top_n dieu luat
    cosim_matrix = util.cos_sim(torch.Tensor(np.array([encoded_ques])), lis_encoded_article)
    relevant_score = cosim_matrix[0]

    # Sap xep theo diem lien quan
    idx_sorted_relevant_score = np.argsort(relevant_score)

    # Du doan lien quan theo top-k
    top_k_aid_sorted = idx_sorted_relevant_score[-top_k:]
    lis_aid_top_k = [arti_pool.article_identity[top_n_aid[i]] for i in top_k_aid_sorted]

    # Du doan lien quan theo threshold
    lis_aid_threshold = [arti_pool.article_identity[top_n_aid[i]]
                         for i, is_greater in enumerate(relevant_score >= threshold) if is_greater]

    # Du doan lien quan theo trail_threshold
    lis_aid_trail_threshold: List[ArticleIdentity] = []
    highest_relevant_score = relevant_score[idx_sorted_relevant_score[-1]]
    for idx in range(len(top_n_aid)):
        if calculate_percent_diff(highest_relevant_score, relevant_score[idx]) <= trail_threshold:
            lis_aid_trail_threshold.append(arti_pool.article_identity[top_n_aid[idx]])

    return lis_aid_threshold, lis_aid_top_k, lis_aid_trail_threshold


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


def calculate_jaccard_sim(u: List[str], v: List[str]):
    co_occurrence = len([iu for iu in u if iu in v])
    return co_occurrence / (len(u) + len(v) - co_occurrence + esp)
