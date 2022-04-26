import os

pkl_root = 'pkl_file'

pkl_article_pool = os.path.join(pkl_root, 'kse_article_pool.pkl')
pkl_question_pool = os.path.join(pkl_root, 'kse_question_pool.pkl')
pkl_private_question_pool = os.path.join(pkl_root, 'private_question_pool.pkl')
pkl_bm25okapi = os.path.join(pkl_root, 'kse_bm25okapi_v1.pkl')
pkl_cached_rel = os.path.join(pkl_root, 'kse_cached_rel_v2_top50.pkl')
pkl_private_cached_rel = os.path.join(pkl_root, 'private_cached_rel_v2_top50.pkl')
pkl_split_ids = os.path.join(pkl_root, 'kse_split_ids.pkl')
