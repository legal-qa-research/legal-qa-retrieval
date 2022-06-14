import pickle
import re

from data_processor.article_pool import ArticlePool
from utils.utilities import get_flat_list_from_preproc, get_raw_from_preproc

if __name__ == '__main__':
    article_pool: ArticlePool = pickle.load(open('pkl_file/alqac_2022_article_pool.pkl', 'rb'))
    cnt_over_256 = 0
    lis_over_256 = []
    for proc_text_pool in article_pool.proc_text_pool:
        lis_token = get_flat_list_from_preproc(proc_text_pool)
        if len(lis_token) >= 256:
            raw_txt = get_raw_from_preproc(proc_text_pool)
            print(raw_txt)
            lis_match = []
            for g in re.finditer(r'\d+\s\.\s', raw_txt):
                lis_match.append((g.start(), g.group()))
            lis_match.append((None, len(raw_txt)))
            for _i in range(len(lis_match) - 1):
                start_pos, group = lis_match[_i]
                nxt_start_pos, nxt_group = lis_match[_i + 1]
                print(group, raw_txt[start_pos:nxt_start_pos])
                if len(raw_txt[start_pos:nxt_start_pos].split(' ')) > 256:
                    lis_over_256.append(len(raw_txt[start_pos:nxt_start_pos].split(' ')))
                    cnt_over_256 += 1
    print('Over 256 tokens: ', cnt_over_256)
    print('Lis length over 256 tokens: ', lis_over_256)
