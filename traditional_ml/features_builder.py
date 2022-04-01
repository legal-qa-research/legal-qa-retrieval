import fasttext


class FeaturesBuilder:
    def __init__(self, fasttext_model=None):
        fasttext_model = fasttext.load_model(fasttext_model)
