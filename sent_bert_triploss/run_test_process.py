from sent_bert_triploss.args_management import args
from sent_bert_triploss.infer_process import InferProcess
from sent_bert_triploss.training_process import TrainingProcess
from data_processor.article_pool import ArticlePool
from data_processor.question_pool import QuestionPool
from bm25_ranking.bm25_ranker import Bm25Ranker
from bm25_ranking.bm25_ranker_cached import Bm25RankerCached

if __name__ == '__main__':
    training_process = InferProcess(args=args)
    training_process.start_test()
