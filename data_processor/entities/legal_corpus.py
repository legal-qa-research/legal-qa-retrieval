import json
from typing import List

from data_processor.entities.law import Law


class LegalCorpus:
    def __init__(self, json_path: str = None, laws: List[Law] = None):
        if json_path is None:
            self.laws = laws
        else:
            json_corpus = json.load(open(json_path, 'r'))
            self.laws = []
            for json_law in json_corpus:
                self.laws.append(Law(json_law=json_law))


if __name__ == '__main__':
    legal_corpus = LegalCorpus(json_path='data/legal_corpus.json')
    pass
