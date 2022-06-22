from typing import List

from data_processor.entities.article_identity import ArticleIdentity


class ArticleRelevantScore:
    article_identity: ArticleIdentity
    relevant_score: float

    def __init__(self, article_identity: ArticleIdentity, relevant_score: float):
        self.article_identity = article_identity
        self.relevant_score = relevant_score


class InferResult:
    qid: str
    list_infer: List[ArticleRelevantScore]

    def __init__(self, qid: str, list_infer: List[ArticleRelevantScore]):
        self.qid = qid
        self.list_infer = list_infer
