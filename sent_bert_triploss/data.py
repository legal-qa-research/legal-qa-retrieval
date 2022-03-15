import pickle

from torch.utils.data import DataLoader

from bm25_ranking.bm25_ranker import Bm25Ranker
from data_processor.article_pool import ArticlePool
from data_processor.question_pool import QuestionPool
from sentence_transformers import InputExample

from sent_bert_triploss.args_management import args
from utils.utilities import split_ids, get_raw_from_preproc


class Data:
    def __init__(self, top_n=100):
        self.top_n = top_n
        self.bm25ranker: Bm25Ranker = pickle.load(open('pkl_file/bm25ranker.pkl', 'rb'))
        self.question_pool: QuestionPool = pickle.load(open('pkl_file/question_pool.pkl', 'rb'))
        self.article_pool: ArticlePool = pickle.load(open('pkl_file/article_pool.pkl', 'rb'))

    def generate_input_examples(self, qid):
        txt_ques = get_raw_from_preproc(self.question_pool.proc_ques_pool[qid])
        candidate_aid = self.bm25ranker.get_topn(ques_id=qid, top_n=self.top_n)
        positive_aid = [self.article_pool.get_position(article_identity) for article_identity in
                        self.question_pool.lis_ques[qid].relevance_articles]
        return [InputExample(texts=[txt_ques, get_raw_from_preproc(self.article_pool.proc_text_pool[aid])],
                             label=int(aid in positive_aid)) for aid in candidate_aid]

    def generate_lis_example(self, lis_qid):
        examples = []
        for qid in lis_qid:
            examples.extend(self.generate_input_examples(qid))
        return examples

    def build_dataset(self, top_n=100):
        lis_train_qid, lis_test_qid = split_ids(n_samples=len(self.question_pool.lis_ques))
        train_examples = self.generate_input_examples(lis_train_qid)
        test_examples = self.generate_input_examples(lis_test_qid)
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
