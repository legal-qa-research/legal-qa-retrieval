class ArticleIdentity:
    def __init__(self, article_identity_json: dict):
        self.law_id = article_identity_json.get('law_id')
        self.article_id = article_identity_json.get('article_id')

    def __eq__(self, other):
        if not isinstance(other, ArticleIdentity):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.law_id == other.law_id and self.article_id == other.article_id
