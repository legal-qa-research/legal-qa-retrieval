from ref_support_bert.ref_sup_data.ref_sup_dataloader import get_ref_sup_dataloader
from ref_support_bert.ref_sup_data.ref_sup_sample import SampleGenerator

from data_processor.article_pool import ArticlePool
from data_processor.question_pool import QuestionPool
from legal_graph.graph_entities.legal_graph import LegalGraph

if __name__ == '__main__':
    sample_generator = SampleGenerator()
    ref_sup_dataloader = get_ref_sup_dataloader(sample_generator.train_examples)
    for data in ref_sup_dataloader:
        print(data)
