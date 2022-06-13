import pickle

from data_processor.article_pool import ArticlePool
from data_processor.entities.article_identity import ArticleIdentity
from legal_graph.graph_entities.legal_graph import LegalGraph
from legal_graph.graph_entities.legal_node import LegalNode
from utils.constant import pkl_article_pool
from utils.utilities import get_raw_from_preproc, get_flat_list_from_preproc
import re
from tqdm import tqdm

key_word = 'điều'


class GraphBuilder:
    def __init__(self):
        self.article_pool: ArticlePool = pickle.load(open(pkl_article_pool, 'rb'))
        self.legal_graph = LegalGraph()
        self.same_law_regex = r'điều\s(\d+)(?:[,\s(?:và)]+(\d+))*\scủa\sbộ_luật\snày'

    def add_all_node(self):
        for aid in self.article_pool.article_identity:
            self.legal_graph.add_node(LegalNode(aid))

    def start_build(self):
        print('Total article: ', len(self.article_pool.proc_text_pool))
        self.add_all_node()
        cnt_refer_article = 0
        cnt_same_law_refer = 0
        for i in range(len(self.article_pool.proc_text_pool)):
            proc_txt = self.article_pool.proc_text_pool[i]
            article_identity = self.article_pool.article_identity[i]
            recent_node = self.legal_graph.get_node(article_identity)
            law_id = self.article_pool.article_identity[i].law_id

            txt_article = get_raw_from_preproc(proc_txt)

            if re.search(r'điều\s(\d+)', txt_article):
                cnt_refer_article += 1

                if re.search(self.same_law_regex, txt_article):
                    cnt_same_law_refer += 1
                    refer_span = re.search(self.same_law_regex, txt_article).group()
                    for same_law_aid in re.findall(r'(\d+)', refer_span):
                        target_node = self.legal_graph.get_node(
                            ArticleIdentity({'law_id': law_id, 'article_id': same_law_aid}))
                        self.legal_graph.add_one_way_vertex(recent_node, target_node)



        print('Refer article: ', cnt_refer_article)
        print('Same law refer article: ', cnt_same_law_refer)


if __name__ == '__main__':
    graph_builder = GraphBuilder()
    graph_builder.start_build()
