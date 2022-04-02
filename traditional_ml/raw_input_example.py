from typing import List


class RawInputExample:
    def __init__(self, ques: List[str], articles: List[str], label: float):
        self.ques = ques
        self.articles = articles
        self.label = label
