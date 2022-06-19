import pickle
from typing import List, Dict, Tuple

from tqdm import tqdm

from data_processor.article_pool import ArticlePool
from data_processor.question_pool import QuestionPool
from legal_graph.graph_entities.legal_graph import LegalGraph
from ref_support_bert.args_management import args
from utils.constant import pkl_question_pool, pkl_article_pool, pkl_split_ids, pkl_cached_rel, pkl_legal_graph
from utils.utilities import get_raw_from_preproc


def get_text(raw_text: str, proc_text: List[List[str]]) -> str:
    return raw_text if args.use_segmenter == 0 else get_raw_from_preproc(proc_text)


class RefSupSample:
    def __init__(self):
        self.query_text: str = ''
        self.lis_article: List[str] = []
        self.label: float = 0.0


class SampleGenerator:
    def __init__(self):
        self.question_pool: QuestionPool = pickle.load(open(pkl_question_pool, 'rb'))
        self.article_pool: ArticlePool = pickle.load(open(pkl_article_pool, 'rb'))
        self.use_segmenter = args.use_segmenter == 1
        self.split_ids_dict: Dict[str, List[int]] = pickle.load(open(pkl_split_ids, 'rb'))
        self.cached_rel = pickle.load(open(pkl_cached_rel, 'rb'))
        self.legal_graph: LegalGraph = pickle.load(open(pkl_legal_graph, 'rb'))

    def __generate_sample_with_qid(self, qid: int, is_train=True) -> List[RefSupSample]:
        # Lay ra text cua cau hoi
        ques_text = get_text(self.question_pool.lis_ques[qid].question, self.question_pool.proc_ques_pool[qid])

        # Lay ra tap cac article lien quan tu BM25
        candidate_aid = self.cached_rel[qid]
        # Lay ra tap cac article thuc su lien quan (gold label)
        positive_aid = [self.article_pool.get_position(article_identity) for article_identity in
                        self.question_pool.lis_ques[qid].relevance_articles]

        # Neu dang tao du lieu de huan luyen mo hinh thi gop ca gold label + ket qua tu BM25
        if is_train:
            candidate_aid = {*candidate_aid, *positive_aid}

        lis_sample: List[RefSupSample] = []
        # Tao sample dua vao tap cac candidate
        for aid in candidate_aid:
            article_identity = self.article_pool.article_identity[aid]
            # Lấy ra đỉnh của đồ thị tương ứng với article đang xét
            recent_node = self.legal_graph.get_node(article_identity)
            # Lấy ra các article được refer bởi article hiện tại
            neighbor_node = recent_node.lis_neighbor
            # Tạo sample bao gồm text của tất cả các article refer và được refer
            sample = RefSupSample()
            sample.query_text = ques_text
            for node in [recent_node, *neighbor_node]:
                node_aid = self.article_pool.get_position(node.identity)
                sample.lis_article.append(
                    get_text(self.article_pool.text_pool[node_aid], self.article_pool.proc_text_pool[node_aid]))
            sample.label = float(aid in positive_aid)
            lis_sample.append(sample)
        return lis_sample

    def __generate_lis_example(self, lis_qid: List[int], is_train: bool = True) -> List[RefSupSample]:
        examples: List[RefSupSample] = []
        for qid in tqdm(lis_qid):
            examples.extend(self.__generate_sample_with_qid(qid, is_train=is_train))
        return examples

    def __split_ids(self):
        if args.is_dev_phase > 0:
            return self.split_ids_dict['train'][:1], self.split_ids_dict['dev']
        else:
            return self.split_ids_dict['train'], self.split_ids_dict['dev']

    def generate_lis_sample(self) -> Tuple[List[RefSupSample], List[RefSupSample]]:
        lis_train_qid, lis_test_qid = self.__split_ids()
        train_examples = self.__generate_lis_example(lis_train_qid, is_train=True)
        test_examples = self.__generate_lis_example(lis_test_qid, is_train=False)
        return train_examples, test_examples


if __name__ == '__main__':
    sample_generator = SampleGenerator()
    lis_train_sample, lis_test_sample = sample_generator.generate_lis_sample()
    pass
