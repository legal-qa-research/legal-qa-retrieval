import json
from tqdm import tqdm

from data_processor.data_segmented import DataSegmented
from utils.utilities import get_raw_from_preproc


def start_making(law_json_path: str):
    law_db = json.load(open(law_json_path, 'r'))
    segmenter = DataSegmented()
    cnt_excep = 0
    with open('data/mini_co_condense_data_segment.txt', 'w') as f:
        for law in tqdm(law_db[:2]):
            for article in law.get('articles'):
                try:
                    f.write(json.dumps({'spans': segmenter.segment_txt(article)}, ensure_ascii=False) + '\n')
                except Exception:
                    print('Exception Caused')
                    cnt_excep += 1
    print('Total exception: ', cnt_excep)


def flatten_segmented_data():
    with open('data/co_condense_data_segment.txt', 'r') as f_in:
        with open('data/co_condense_data_segment_raw.txt', 'w') as f_out:
            for line in tqdm(f_in.readlines()):
                segmented_lis = json.loads(line)['spans']
                raw = get_raw_from_preproc(segmented_lis)
                f_out.write(json.dumps({'spans': raw}, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    start_making('data/concat_law.json')
    # flatten_segmented_data()
