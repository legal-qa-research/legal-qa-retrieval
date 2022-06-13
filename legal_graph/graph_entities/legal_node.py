from typing import List

from data_processor.entities.article_identity import ArticleIdentity


class LegalNode:
    def __init__(self, identity: ArticleIdentity):
        self.identity = identity
        self.lis_neighbor: List[LegalNode] = []

    def add_neighbor(self, neighbor_identity: 'LegalNode'):
        assert isinstance(neighbor_identity, LegalNode), 'The input neighbor is not Legal Node class'
        self.lis_neighbor.append(neighbor_identity)
