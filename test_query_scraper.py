import requests, json, bs4
from progress.bar import *
import pandas as pd
from urllib.parse import ParseResult, urlparse
import crayons
from sklearn.utils import shuffle

set_N = 200
def df_preproc(dfm):
    dfm = dfm.drop(["source_domain", "tweet_num"], axis=1)
    dfm = dfm.loc[:, ~dfm.columns.str.contains('^Unnamed')]
    dfm = dfm.dropna(subset=['news_url'])
    return dfm

df_test = pd.read_csv("data/orig/FakeNewsNet.csv")
df_test.drop_duplicates(inplace = True)
df_test.dropna(inplace = True)
df_test = shuffle(df_test)

status = {
    'success_real': 0,
    'success_fake': 0,
    'failed': 0,
    'tested': 0
}

with PixelBar('Scraping page...', max=set_N) as bar:
    for idx, row in df_test.sample(n=set_N).iterrows():
        bar.bar_prefix = 'Scraping "{}"...'.format(str(row["title"])[:15])
        bar.update()

        p = urlparse(row["news_url"], 'https')
        netloc = p.netloc or p.path
        path = p.path if p.netloc else ''
        if not netloc.startswith('www.'):
            netloc = 'www.' + netloc
        fixed_url = ParseResult('https', netloc, path, *p[3:]).geturl()

        try:
            response = requests.get(
                fixed_url, headers={'User-Agent': 'Mozilla/5.0', 'cache-control': 'max-age=0'}, cookies={'cookies': ''})
            soup = bs4.BeautifulSoup(response.text, features="html.parser")

            article_text = soup.article.get_text(' ', strip=True)
            print("\n{}\n".format(article_text))

            r = requests.post("http://127.0.0.1:8003/classify", json={"text": article_text})
            results = r.json()

            if row["real"] - results["weighted_avg"] > 0.5 or row["real"] - results["weighted_avg"] < -0.5:
                status["failed"] += 1
            else:
                if row["real"] == 1:
                    status["success_real"] += 1
                else:
                    status["success_fake"] += 1

            print("Truth Results for {}: \n{}\n".format(fixed_url, json.dumps(results, indent=2)))
        except Exception as e:
            print("Error: ", end='')
            print(e)

        status["tested"] += 1
        bar.next()

print(json.dumps(status, indent=2))