from typing import List

from data_processor.entities.article_identity import ArticleIdentity
from legal_graph.graph_entities.legal_node import LegalNode


class LegalGraph:
    def __init__(self):
        self.lis_node: List[LegalNode] = []

    def add_node(self, node: LegalNode):
        self.lis_node.append(node)

    def get_node(self, aid: ArticleIdentity) -> LegalNode:
        exist_node: List[LegalNode] = [node for node in self.lis_node if node.identity == aid]
        assert len(exist_node) == 1, f'Number of node exist is {len(exist_node)}, which is not reasonable. It must be 1'
        return exist_node[0]

    def add_one_way_vertex(self, node_1: LegalNode, node_2: LegalNode):
        exist_node: List[LegalNode] = [node for node in self.lis_node if node == node_1 or node == node_2]
        assert len(exist_node) == 2, f'Number of exist node is {len(exist_node)}, which is not reasonable. It must be 2'
        node_1.add_neighbor(node_2)

    def count_vertex(self):
        cnt_vertex = 0
        for node in self.lis_node:
            cnt_vertex += len(node.lis_neighbor)
        return cnt_vertex
