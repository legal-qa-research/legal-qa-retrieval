import json

from tqdm import tqdm


def start_generate_from_crawl_data():
    concat_law = json.load(open('data/concat_law.json', 'r'))
    print(len([article for law in concat_law for article in law.get('articles')]))
    with open('data/crawled_legal_corpus.txt', 'w') as f:
        for law in tqdm(concat_law):
            for article in law.get('articles'):
                f.write(article + '\n')


def start_generate_from_competition_data():
    law_database = json.load(open('data/legal_corpus.json', 'r'))
    print(len([article for law in law_database for article in law.get('articles')]))
    with open('data/competition_legal_corpus.txt', 'w') as f:
        for law in tqdm(law_database):
            for article in law.get('articles'):
                f.write(str(article.get('text')).replace('\n', ' ') + '\n')


if __name__ == '__main__':
    # start_generate_from_crawl_data()
    start_generate_from_competition_data()
