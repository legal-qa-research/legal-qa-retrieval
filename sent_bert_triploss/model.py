from sentence_transformers import SentenceTransformer, models
from sent_bert_triploss.args_management import args

bert_embedding_model = models.Transformer(args.model_name, max_seq_length=args.max_seq_len)

pooling_model = models.Pooling(word_embedding_dimension=bert_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_cls_token=True)

model = SentenceTransformer(modules=[bert_embedding_model, pooling_model])
