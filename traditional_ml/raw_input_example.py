from typing import List


class RawInputExample:
    prob: float = None

    def __init__(self, ques_id: int, ques: List[str], article_id: int, articles: List[str], label: float):
        self.ques_id = ques_id
        self.article_id = article_id
        self.ques = ques
        self.articles = articles
        self.label = label
