from sent_bert_triploss.args_management import args
from sent_bert_triploss.training_process import TrainingProcess
from data_processor.article_pool import ArticlePool
from data_processor.question_pool import QuestionPool
from bm25_ranking.bm25_ranker import Bm25Ranker

if __name__ == '__main__':
    training_process = TrainingProcess(args=args)
    training_process.start_training()
