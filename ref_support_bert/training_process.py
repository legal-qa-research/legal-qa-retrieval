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
    device = 'cpu'
    gpus = None
    auto_select_gpus = None
    if torch.cuda.is_available():
        device = 'gpu'
        gpus = args.n_gpus
        auto_select_gpus = True
    model = RefSupModel(input_size=args.embed_size, args=args)
    trainer = Trainer(accelerator=device, max_epochs=args.n_epochs, default_root_dir=args.root_dir, gpus=gpus,
                      auto_select_gpus=auto_select_gpus, log_every_n_steps=2)
    trainer.fit(model=model)

    pass
