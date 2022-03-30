import os.path
import pickle
from random import shuffle
from typing import List, Tuple, Dict

from torch.utils.data import DataLoader

from data_processor.article_pool import ArticlePool
from data_processor.question_pool import QuestionPool
from sentence_transformers import InputExample

from sent_bert_triploss.constant import pkl_split_ids
from utils.utilities import get_raw_from_preproc


class Data:
    def __init__(self, pkl_question_pool_path: str, pkl_article_pool_path: str, pkl_cached_rel_path: str,
                 pkl_cached_split_ids: str, args):
        self.cached_rel = pickle.load(open(pkl_cached_rel_path, 'rb'))
        self.question_pool: QuestionPool = pickle.load(open(pkl_question_pool_path, 'rb'))
        self.article_pool: ArticlePool = pickle.load(open(pkl_article_pool_path, 'rb'))
        self.split_ids_dict = None
        self.args = args
        if pkl_cached_split_ids is not None and os.path.exists(pkl_cached_split_ids):
            self.split_ids_dict: Dict[str, List[int]] = pickle.load(open(pkl_cached_split_ids, 'rb'))

    def generate_input_examples(self, qid: int, is_train: bool = True) -> List[InputExample]:
        txt_ques = get_raw_from_preproc(self.question_pool.proc_ques_pool[qid])
        candidate_aid = self.cached_rel[qid]
        positive_aid = [self.article_pool.get_position(article_identity) for article_identity in
                        self.question_pool.lis_ques[qid].relevance_articles]
        if is_train:
            candidate_aid = {*candidate_aid, *positive_aid}

        return [InputExample(texts=[txt_ques, get_raw_from_preproc(self.article_pool.proc_text_pool[aid])],
                             label=int(aid in positive_aid)) for aid in candidate_aid]

    def generate_lis_example(self, lis_qid, is_train: bool = True):
        examples = []
        for qid in lis_qid:
            examples.extend(self.generate_input_examples(qid, is_train=is_train))
        return examples

    def split_ids(self, test_size=0.2):
        n_samples = len(self.question_pool.lis_ques)
        if self.split_ids_dict is None:
            train_size = 1 - test_size
            cut_pos = int(n_samples * train_size)
            lis_id = [i for i in range(n_samples)]
            shuffle(lis_id)
            self.split_ids_dict = {
                'train': lis_id[:cut_pos],
                'dev': lis_id[cut_pos:]
            }
            pickle.dump(self.split_ids_dict, open(pkl_split_ids, 'wb'))
        if self.args.is_dev_phase > 0:
            return self.split_ids_dict['train'][:1], self.split_ids_dict['dev'][:1]
        else:
            return self.split_ids_dict['train'], self.split_ids_dict['dev']

    def build_dataset(self) -> Tuple[DataLoader, List[InputExample]]:
        lis_train_qid, lis_test_qid = self.split_ids()
        train_examples = self.generate_lis_example(lis_train_qid, is_train=True)
        test_examples = self.generate_lis_example(lis_test_qid, is_train=False)
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=self.args.batch_size)
        # test_dataloader = DataLoader(test_examples, shuffle=False, batch_size=args.batch_size)
        return train_dataloader, test_examples
