import os

from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, AutoConfig, BatchEncoding
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaConfig, \
    RobertaClassificationHead, SequenceClassifierOutput

from transformers.models.phobert import PhobertTokenizer
import torch


class RefSupRoberta(nn.Module):
    def __init__(
            self,
            roberta: RobertaModel,
            config: RobertaConfig
    ):
        super(RefSupRoberta, self).__init__()

        self.num_labels = config.num_labels
        self.config = config

        self.lm = roberta
        self.classifier = RobertaClassificationHead(config)

    def forward(self, model_input: BatchEncoding, labels: Tensor):
        outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.lm(
            **model_input,
            output_hidden_states=True,
            return_dict=True
        )

        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(input=logits.view(-1, self.num_labels), target=labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

    def save_pretrained(self, output_dir: str):
        self.lm.save_pretrained(output_dir)
        model_dict = self.state_dict()
        torch.save(model_dict, os.path.join(output_dir, 'model.pt'))
        torch.save([self.data_args, self.model_args, self.train_args], os.path.join(output_dir, 'args.pt'))


if __name__ == '__main__':
    model_name = 'vinai/phobert-base'
    core_model = AutoModel.from_pretrained(model_name)
    core_config = AutoConfig.from_pretrained(model_name)
    core_config.num_labels = 2
    tokenizer = PhobertTokenizer.from_pretrained(model_name)

    model = RefSupRoberta(roberta=core_model, config=core_config)

    inp = tokenizer('tôi là sinh_viên trường Đại học Công Nghệ.', padding='max_length', return_tensors='pt')
    output_model = model.forward(model_input=inp, labels=torch.tensor([1]))
    pass
