import os

pkl_root = 'pkl_file'
data_root = os.path.join('data', 'ALQAC_2022')

json_train_question = os.path.join(data_root, 'question_clean_v2.json')
json_law = os.path.join(data_root, 'law.json')
json_test_question = os.path.join(data_root, 'ALQAC_test_release.json')

pkl_article_pool = os.path.join(pkl_root, 'alqac_2022_article_pool.pkl')
pkl_question_pool = os.path.join(pkl_root, 'alqac_2022_question_pool.pkl')
pkl_test_question_pool = os.path.join(pkl_root, 'alqac_2022_test_question_pool.pkl')
pkl_bm25okapi = os.path.join(pkl_root, 'alqac_2022_bm25okapi_v1.pkl')
pkl_cached_rel = os.path.join(pkl_root, 'alqac_2022_cached_rel_v1_top2000.pkl')
pkl_private_cached_rel = os.path.join(pkl_root, 'alqac_2022_test_cached_rel_v1_top2000.pkl')
pkl_split_ids = os.path.join(pkl_root, 'alqac_2022_split_ids.pkl')
pkl_tfidf = os.path.join(pkl_root, 'alqac_2022_tfidf.pkl')
pkl_legal_graph = os.path.join(pkl_root, 'alqac_2022_legal_graph.pkl')
pkl_xgb_model = os.path.join(pkl_root, 'alqac_2022_xgb_model.pkl')
xgb_model = os.path.join(pkl_root, 'alqac_2022_xgb_model.json')
pkl_bm25_infer_result = os.path.join(pkl_root, 'alqac_2022_infer_bm25_private_test.pkl')
pkl_sent_bert_v4_infer_result = os.path.join(pkl_root, 'alqac_2022_infer_test_sent_bert_v4_full_v1_private_test.pkl')
pkl_sent_bert_v7_infer_result = os.path.join(pkl_root, 'alqac_2022_infer_test_sent_bert_v7_full_v1_private_test.pkl')
pkl_xgboost_infer_result = os.path.join(pkl_root, 'alqac_2022_infer_xgboost_private_test.pkl')
ensemble_log = os.path.join('ensemble', 'ensemble_log.txt')
