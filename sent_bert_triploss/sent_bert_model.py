from sentence_transformers import SentenceTransformer, models

from sent_bert_triploss.args_management import args


def get_sent_bert_model(load_chk_point_path) -> SentenceTransformer:
    if load_chk_point_path is not None:
        print(f'Load model from {load_chk_point_path}')
        return SentenceTransformer(model_name_or_path=load_chk_point_path)

    print('Init model from BERT checkpoint')
    bert_embedding_model = models.Transformer(args.model_name, max_seq_length=args.max_seq_len)

    pooling_model = models.Pooling(word_embedding_dimension=bert_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_cls_token=True)

    return SentenceTransformer(modules=[bert_embedding_model, pooling_model])
