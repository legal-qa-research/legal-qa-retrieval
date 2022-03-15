class ArticleIdentity:
    def __init__(self, article_identity_json: dict):
        self.law_id = article_identity_json.get('law_id')
        self.article_id = article_identity_json.get('article_id')
