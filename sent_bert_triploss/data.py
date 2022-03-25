import pickle
from typing import List, Tuple

from torch.utils.data import DataLoader

from data_processor.article_pool import ArticlePool
from data_processor.question_pool import QuestionPool
from sentence_transformers import InputExample

from sent_bert_triploss.args_management import args
from utils.utilities import split_ids, get_raw_from_preproc


class Data:
    def __init__(self, pkl_question_pool_path: str, pkl_article_pool_path: str, pkl_cached_rel_path: str):
        self.cached_rel = pickle.load(open(pkl_cached_rel_path, 'rb'))
        self.question_pool: QuestionPool = pickle.load(open(pkl_question_pool_path, 'rb'))
        self.article_pool: ArticlePool = pickle.load(open(pkl_article_pool_path, 'rb'))

    def generate_input_examples(self, qid: int) -> List[InputExample]:
        txt_ques = get_raw_from_preproc(self.question_pool.proc_ques_pool[qid])
        candidate_aid = self.cached_rel[qid]
        positive_aid = [self.article_pool.get_position(article_identity) for article_identity in
                        self.question_pool.lis_ques[qid].relevance_articles]
        candidate_aid = {*candidate_aid, *positive_aid}
        return [InputExample(texts=[txt_ques, get_raw_from_preproc(self.article_pool.proc_text_pool[aid])],
                             label=int(aid in positive_aid)) for aid in candidate_aid]

    def generate_lis_example(self, lis_qid):
        examples = []
        for qid in lis_qid:
            examples.extend(self.generate_input_examples(qid))
        return examples

    def build_dataset(self) -> Tuple[DataLoader, List[InputExample]]:
        lis_train_qid, lis_test_qid = split_ids(n_samples=len(self.question_pool.lis_ques))
        # lis_train_qid, lis_test_qid = split_ids(n_samples=2)
        train_examples = self.generate_lis_example(lis_train_qid)
        test_examples = self.generate_lis_example(lis_test_qid)
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
        # test_dataloader = DataLoader(test_examples, shuffle=False, batch_size=args.batch_size)
        return train_dataloader, test_examples
