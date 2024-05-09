import re, string, json, time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump, load
from pathlib import Path
from flask import Flask, jsonify, request
from progress.bar import *
from progress.spinner import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import requests, json, bs4
from progress.bar import *
import pandas as pd
from urllib.parse import ParseResult, urlparse
import crayons
from sklearn.utils import shuffle





df_test = pd.read_csv("data/orig/FakeNewsNet.csv")
df_test.drop_duplicates(inplace = True)
df_test.dropna(inplace = True)
#df_test = shuffle(df_test)

df = []

set_N = 100

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
            #print("\n{}\n".format(article_text))

            df.append([row["real"], article_text])

        except Exception as e:
            print("Error: ", end='')
            print(e)

        bar.next()

df = pd.DataFrame(df)
df = df.rename(columns={0: 'label', 1: 'text'})

df.to_csv('data/FNNScraped.csv')