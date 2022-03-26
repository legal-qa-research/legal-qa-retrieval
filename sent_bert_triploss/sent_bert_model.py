from sentence_transformers import SentenceTransformer, models

from sent_bert_triploss.args_management import args

train_layers = [
    'auto_model.pooler.dense.bias',
    'auto_model.pooler.dense.weight'
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
