import torch.cuda
from pytorch_lightning import Trainer

from ref_support_bert.args_management import args
from ref_support_bert.ref_sup_modeling.modeling import RefSupModel

if __name__ == '__main__':
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    model = RefSupModel(input_size=768 * 2, args=args)
    trainer = Trainer(accelerator=device, max_epochs=20, default_root_dir=args.root_dir)
    trainer.fit(model=model)
