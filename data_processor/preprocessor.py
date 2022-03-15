from data_processor.data_segmented import DataSegmented


class Preprocessor:
    def __init__(self):
        self.__segmenter = DataSegmented()

    def preprocess(self, _txt_article: str):
        _txt_article = _txt_article.lower()
        return self.__segmenter.segment_txt(_txt_article)
