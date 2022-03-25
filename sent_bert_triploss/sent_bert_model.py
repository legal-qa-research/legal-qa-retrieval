from sentence_transformers import SentenceTransformer, models

from sent_bert_triploss.args_management import args

train_layers = [
    'auto_model.pooler.dense.bias',
    'auto_model.pooler.dense.weight',
    'auto_model.encoder.layer.11.attention.self.query.weight',
    'auto_model.encoder.layer.11.attention.self.query.bias',
    'auto_model.encoder.layer.11.attention.self.key.weight',
    'auto_model.encoder.layer.11.attention.self.key.bias',
    'auto_model.encoder.layer.11.attention.self.value.weight',
    'auto_model.encoder.layer.11.attention.self.value.bias',
    'auto_model.encoder.layer.11.attention.output.dense.weight',
    'auto_model.encoder.layer.11.attention.output.dense.bias',
    'auto_model.encoder.layer.11.attention.output.LayerNorm.weight',
    'auto_model.encoder.layer.11.attention.output.LayerNorm.bias',
    'auto_model.encoder.layer.11.intermediate.dense.weight',
    'auto_model.encoder.layer.11.intermediate.dense.bias',
    'auto_model.encoder.layer.11.output.dense.weight',
    'auto_model.encoder.layer.11.output.dense.bias',
    'auto_model.encoder.layer.11.output.LayerNorm.weight',
    'auto_model.encoder.layer.11.output.LayerNorm.bias'
]


def get_sent_bert_model() -> SentenceTransformer:
    bert_embedding_model = models.Transformer(args.model_name, max_seq_length=args.max_seq_len)
    for name, params in bert_embedding_model.named_parameters():
        if name in train_layers:
            params.requires_grad = True
        else:
            params.requires_grad = False

    pooling_model = models.Pooling(word_embedding_dimension=bert_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_cls_token=True)

    sent_transformer_model = SentenceTransformer(modules=[bert_embedding_model, pooling_model])
    return sent_transformer_model
