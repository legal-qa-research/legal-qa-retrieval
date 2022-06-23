import os

pkl_root = 'pkl_file'

pkl_article_pool = os.path.join(pkl_root, 'alqac_2022_article_pool.pkl')
pkl_question_pool = os.path.join(pkl_root, 'alqac_2022_question_pool.pkl')
pkl_test_question_pool = os.path.join(pkl_root, 'alqac_2022_test_question_pool.pkl')
pkl_bm25okapi = os.path.join(pkl_root, 'alqac_2022_bm25okapi_v1.pkl')
pkl_cached_rel = os.path.join(pkl_root, 'alqac_2022_cached_rel_v1_top100.pkl')
pkl_private_cached_rel = os.path.join(pkl_root, 'alqac_2022_test_cached_rel_v1_top2000.pkl')
pkl_split_ids = os.path.join(pkl_root, 'alqac_2022_split_ids.pkl')
pkl_tfidf = os.path.join(pkl_root, 'alqac_2022_tfidf.pkl')
pkl_legal_graph = os.path.join(pkl_root, 'alqac_2022_legal_graph.pkl')
pkl_xgb_model = os.path.join(pkl_root, 'alqac_2022_xgb_model.pkl')
