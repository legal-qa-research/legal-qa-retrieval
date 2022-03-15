class Article:
    def __init__(self, json_article: dict = None, article_id: str = None, title: str = None, text: str = None):
        if json_article is None:
            self.article_id = article_id
            self.title = title
            self.text = text
        else:
            self.article_id = json_article.get('article_id')
            self.title = json_article.get('title')
            self.text = json_article.get('text')
