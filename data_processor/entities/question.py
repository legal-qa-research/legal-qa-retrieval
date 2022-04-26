from typing import List

from data_processor.entities.article_identity import ArticleIdentity


class Question:
    relevance_articles: List[ArticleIdentity]
    question_id: str
    question: str

    def __init__(self, question_json: dict):
        self.question_id = question_json.get('question_id')
        self.question = question_json.get('text')
        if question_json.get('relevant_articles') is not None:
            self.relevance_articles = [ArticleIdentity(article_identity_json)
                                       for article_identity_json in question_json.get('relevant_articles')]
        else:
            self.relevance_articles = []
