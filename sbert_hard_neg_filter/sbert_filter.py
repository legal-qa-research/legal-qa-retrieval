import pickle
from typing import List

import torch
from sentence_transformers import SentenceTransformer, util, InputExample
from tqdm.autonotebook import tqdm

from sbert_hard_neg_filter.constant import hard_neg_path
from sent_bert_triploss.data import Data
from utils.constant import pkl_question_pool, pkl_article_pool, pkl_cached_rel


class SBertFilter:
    def __init__(self, args):
        self.args = args
        self.model = SentenceTransformer(model_name_or_path=args.model_name_or_path)
        self.model.eval()
        self.data = Data(pkl_question_pool_path=pkl_question_pool, pkl_article_pool_path=pkl_article_pool,
                         pkl_cached_rel_path=pkl_cached_rel, pkl_cached_split_ids=self.args.split_ids,
                         args=args)

    def filer_hard_negative_for_single_qid(self, qid: int) -> List[InputExample]:
        lis_input_example = self.data.generate_input_examples(qid=qid, is_train=True)
        assert len(lis_input_example) > 0, 'List input example is empty'
        positive_articles = [example.texts[1] for example in lis_input_example if example.label == 1]
        negative_articles = [example.texts[1] for example in lis_input_example if example.label == 0]
        txt_ques = lis_input_example[0].texts[0]
        encoded_ques = self.model.encode([txt_ques])
        encoded_pos_articles = self.model.encode(positive_articles)
        encoded_neg_articles = self.model.encode(negative_articles)
        pos_examples_score = util.cos_sim(encoded_ques, encoded_pos_articles)
        min_pos_score = torch.min(pos_examples_score[0])
        neg_examples_score = util.cos_sim(encoded_ques, encoded_neg_articles)
        lis_hard_neg_ids = [i for i in range(len(negative_articles)) if neg_examples_score[0, i] >= min_pos_score]
        hard_neg_articles = [negative_articles[i] for i in lis_hard_neg_ids]
        return [InputExample(texts=[txt_ques, txt_article], label=0.0) for txt_article in hard_neg_articles] \
               + [InputExample(texts=[txt_ques, txt_article], label=1.0) for txt_article in positive_articles]

    def start_filter_negative_pair(self):
        lis_train_qid, lis_test_qid = self.data.split_ids()
        lis_r2_example = []
        last_len = 0
        for qid in tqdm(lis_train_qid):
            lis_r2_example.extend(self.filer_hard_negative_for_single_qid(qid))
            if len(lis_r2_example) - last_len >= 100:
                last_len = len(lis_r2_example)
                print(last_len)

        pickle.dump(lis_r2_example, open(hard_neg_path, 'wb'))
