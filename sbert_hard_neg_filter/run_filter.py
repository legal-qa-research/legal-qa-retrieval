from sbert_hard_neg_filter.args_management import args
from sbert_hard_neg_filter.sbert_filter import SBertFilter
from data_processor.article_pool import ArticlePool
from data_processor.question_pool import QuestionPool

if __name__ == '__main__':
    filter_process = SBertFilter(args=args)
    filter_process.start_filter_negative_pair()
