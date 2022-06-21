import torch.cuda
from pytorch_lightning import Trainer

from ref_support_bert.args_management import args
from ref_support_bert.ref_sup_data.ref_sup_dataloader import get_ref_sup_dataloader
from ref_support_bert.ref_sup_data.ref_sup_sample import SampleGenerator
from ref_support_bert.ref_sup_modeling.modeling import RefSupModel

from data_processor.article_pool import ArticlePool
from data_processor.question_pool import QuestionPool
from legal_graph.graph_entities.legal_graph import LegalGraph

if __name__ == '__main__':
    sample_generator = SampleGenerator()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RefSupModel(input_size=768 * 2, args=args)
    trainer = Trainer(accelerator=device, max_epochs=20, default_root_dir=args.root_dir)
    trainer.fit(model=model, train_dataloader=get_ref_sup_dataloader(sample_generator.train_examples),
                val_dataloaders=[get_ref_sup_dataloader(test_batch) for test_batch in sample_generator.test_examples])

    pass
