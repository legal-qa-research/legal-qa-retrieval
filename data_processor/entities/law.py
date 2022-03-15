from typing import List

from data_processor.entities.article import Article


class Law:
    def __init__(self, json_law: dict = None, law_id: str = None, articles: List[Article] = None):
        if json_law is None:
            self.law_id = law_id
            self.articles = articles
        else:
            self.law_id = json_law.get('law_id')
            self.articles = []
            for json_article in json_law.get('articles'):
                self.articles.append(Article(json_article=json_article))
