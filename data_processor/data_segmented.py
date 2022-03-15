from vncorenlp import VnCoreNLP


class DataSegmented:
    def __init__(self):
        self.__vncorenlp = VnCoreNLP("/Users/LongNH/Tools/VnCoreNLP/VnCoreNLP-1.1.1.jar", annotators="wseg",
                                     max_heap_size='-Xmx2g')
        self.__max_len = 10000

    def segment_txt(self, _txt_article: str):
        num_token = len(_txt_article.split())
        lis_token = _txt_article.split()
        res_seq = []
        for i in range(num_token // self.__max_len + 1):
            sub_lis_token = lis_token[i * self.__max_len:min((i + 1) * self.__max_len, num_token)]
            segmented_token = self.__vncorenlp.tokenize(' '.join(sub_lis_token))
            res_seq.extend(segmented_token)
        return res_seq
