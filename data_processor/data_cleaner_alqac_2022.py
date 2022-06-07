import json
import re

if __name__ == '__main__':
    ques_json = json.load(open('data/ALQAC_2022/question.json', 'r'))
    cnt_err = 0
    for ques in ques_json:
        for rel_article in ques.get('relevant_articles'):
            if not rel_article.get('article_id').isnumeric():
                cnt_err += 1
                err_article_id = rel_article.get('article_id')
                regex_obj_search = re.search(r'Điều\s(\d+)\.', err_article_id)
                update_article_id = regex_obj_search.group(1).strip()
                rel_article['article_id'] = update_article_id

    json.dump(ques_json, open('data/ALQAC_2022/question_clean_v1.json', 'w'), ensure_ascii=False)
    print(cnt_err)
