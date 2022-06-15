from typing import List


class RefSupSample:
    def __init__(self):
        self.query_text: str = ''
        self.lis_article: List[str] = []
        self.label: float = 0.0


def create_lis_sample() -> List[RefSupSample]:
    ref_sup_sample = RefSupSample()
    ref_sup_sample.query_text = 'Đây là câu hỏi'
    ref_sup_sample.lis_article = ['Đây là câu trả lời', 'Đây là câu trả lời']
    return [ref_sup_sample]
