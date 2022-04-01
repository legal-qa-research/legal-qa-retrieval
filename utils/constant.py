import os

pkl_root = 'pkl_file'

pkl_article_pool = os.path.join(pkl_root, 'article_pool.pkl')
pkl_question_pool = os.path.join(pkl_root, 'question_pool.pkl')
pkl_private_question_pool = os.path.join(pkl_root, 'private_question_pool.pkl')
pkl_bm25okapi = os.path.join(pkl_root, 'bm25okapi.pkl')
pkl_cached_rel = os.path.join(pkl_root, 'cached_rel_v2_top50.pkl')
pkl_private_cached_rel = os.path.join(pkl_root, 'private_cached_rel_v2_top50.pkl')
pkl_split_ids = os.path.join(pkl_root, 'split_ids.pkl')
