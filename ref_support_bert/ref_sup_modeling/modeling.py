import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
from torch import nn
from torch.nn import GRU
from torch.nn.utils.rnn import PackedSequence
from pytorch_lightning.core.lightning import LightningModule
from torch.nn.functional import cross_entropy

from ref_support_bert.args_management import RefSupArgument
from transformers.optimization import AdamW


class RefSupModel(LightningModule):
    def __init__(self, input_size, args: RefSupArgument):
        super(RefSupModel, self).__init__()
        self.args = args
        self.recurrent_layer = GRU(input_size=input_size, hidden_size=input_size, num_layers=2)
        self.linear_relu_stack = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 2)
        )

    def forward(self, model_input: PackedSequence):
        gru_output, gru_hidden = self.recurrent_layer.forward(input=model_input)
        embed_output = gru_output.data[-1]
        output = self.linear_relu_stack.forward(embed_output)
        return output

    def training_step(self, sample, sample_idx) -> STEP_OUTPUT:
        model_input, label = sample
        model_predict = self.forward(model_input)
        model_predict = torch.unsqueeze(model_predict, dim=0)
        label_vec = torch.as_tensor([[float(label == 0), float(label == 1)]], dtype=torch.float)
        loss = cross_entropy(input=model_predict, target=label_vec)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return AdamW(params=self.parameters(), lr=self.args.lr)
