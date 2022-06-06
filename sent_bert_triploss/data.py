import os.path
import pickle
from random import shuffle
from typing import List, Tuple, Dict

from torch.utils.data import DataLoader

from data_processor.article_pool import ArticlePool
from data_processor.question_pool import QuestionPool
from sentence_transformers import InputExample

from utils.constant import pkl_split_ids
from utils.utilities import get_raw_from_preproc


class Data:
    def __init__(self, pkl_question_pool_path: str, pkl_article_pool_path: str, pkl_cached_rel_path: str,
                 pkl_cached_split_ids: str, args):
        assert pkl_cached_split_ids is not None and os.path.exists(pkl_cached_split_ids), 'Split ids is not exist'
        self.cached_rel = pickle.load(open(pkl_cached_rel_path, 'rb'))
        self.question_pool: QuestionPool = pickle.load(open(pkl_question_pool_path, 'rb'))
        self.article_pool: ArticlePool = pickle.load(open(pkl_article_pool_path, 'rb'))
        self.args = args
        self.use_segmenter = self.args.use_segmenter == 1
        self.split_ids_dict: Dict[str, List[int]] = pickle.load(open(pkl_cached_split_ids, 'rb'))

    def generate_input_examples(self, qid: int, is_train: bool = True) -> List[InputExample]:
        if self.use_segmenter:
            txt_ques = get_raw_from_preproc(self.question_pool.proc_ques_pool[qid])
        else:
            txt_ques = self.question_pool.lis_ques[qid].question
        candidate_aid = self.cached_rel[qid]
        positive_aid = [self.article_pool.get_position(article_identity) for article_identity in
                        self.question_pool.lis_ques[qid].relevance_articles]
        if is_train:
            candidate_aid = {*candidate_aid, *positive_aid}

        return [InputExample(texts=[
            txt_ques,
            get_raw_from_preproc(self.article_pool.proc_text_pool[aid])
            if self.use_segmenter else self.article_pool.text_pool[aid]
        ],
            label=float(aid in positive_aid)) for aid in candidate_aid]

    def generate_lis_example(self, lis_qid, is_train: bool = True) -> List[InputExample]:
        examples = []
        for qid in lis_qid:
            examples.extend(self.generate_input_examples(qid, is_train=is_train))
        return examples

    def split_ids(self):
        if self.args.is_dev_phase > 0:
            return self.split_ids_dict['train'][:1], self.split_ids_dict['dev']
        else:
            return self.split_ids_dict['train'], self.split_ids_dict['dev']

    def build_dataset(self) -> Tuple[DataLoader, List[InputExample]]:
        lis_train_qid, lis_test_qid = self.split_ids()
        train_examples = self.generate_lis_example(lis_train_qid, is_train=True)
        test_examples = self.generate_lis_example(lis_test_qid, is_train=False)
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=self.args.batch_size)
        # test_dataloader = DataLoader(test_examples, shuffle=False, batch_size=args.batch_size)
        return train_dataloader, test_examples
