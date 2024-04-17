import re, string, json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump, load
from pathlib import Path

from progress.bar import *
from progress.spinner import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

# Load data sets
Path("data/models").mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/train_nobert.csv")
df_test = pd.read_csv("data/test_nobert.csv")

# Data clean
def df_preproc(dfm):
    dfm = dfm.drop(["title", ], axis=1)
    dfm = dfm.loc[:, ~dfm.columns.str.contains('^Unnamed')]
    dfm = dfm.dropna(subset=['text'])
    dfm = dfm[dfm["text"].str.strip() != ""]
    return dfm

# Word cleaning
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Preproc and split data
with PixelBar('', max=6) as bar:
    bar.bar_prefix = 'Preprocessing data...'
    bar.update()

    # Preprocess text and datasets
    df = df_preproc(df)
    df_test = df_preproc(df_test)

    df["text"] = df["text"].apply(wordopt)
    df_test["text"] = df_test["text"].apply(wordopt)

    x_train = df["text"]
    y_train = df["label"]
    x_test = df_test["text"]
    y_test = df_test["label"]

    bar.bar_prefix = 'Vectorizing text...'
    bar.next()

    # Vectorize text
    vectorization = TfidfVectorizer()
    xv_train = vectorization.fit_transform(x_train)
    xv_test = vectorization.transform(x_test)
    """dump(xv_train, 'data/models/xv_train.joblib')"""

    bar.bar_prefix = 'Training logistic regression model...'
    bar.next()

    # Logistic regression
    LR = LogisticRegression()
    LR.fit(xv_train,y_train)
    dump(LR, 'data/models/lr.joblib')
    pred_lr=LR.predict(xv_test)

    bar.bar_prefix = 'Training decision tree model...'
    bar.next()

    # DT Classifier
    DT = DecisionTreeClassifier()
    DT.fit(xv_train, y_train)
    dump(DT, 'data/models/dt.joblib')
    pred_dt = DT.predict(xv_test)

    bar.bar_prefix = 'Training gradient boosting model...'
    bar.next()

    # GB Classifier
    GBC = GradientBoostingClassifier(random_state=0)
    GBC.fit(xv_train, y_train)
    dump(GBC, 'data/models/gbc.joblib')
    pred_gbc = GBC.predict(xv_test)

    bar.bar_prefix = 'Training random forest model...'
    bar.next()

    # RF Classifier
    RFC = RandomForestClassifier(random_state=0)
    RFC.fit(xv_train, y_train)
    RandomForestClassifier(random_state=0)
    dump(RFC, 'data/models/rfc.joblib')
    pred_rfc = RFC.predict(xv_test)

    bar.bar_prefix = 'Done'
    bar.next()

    # Save score weights
    scores = {
        "score_lr": LR.score(xv_test, y_test),
        "score_dt": DT.score(xv_test, y_test),
        "score_gbc": GBC.score(xv_test, y_test),
        "score_rfc": RFC.score(xv_test, y_test)
    }
    json.dump(scores, open('data/models/score_weights.json', mode='w'))

    # Results
    print("\n\nLogistic regression prediction score: {}\n".format(scores["score_gbc"]))

    print("DTC prediction score: {}".format(scores["score_dt"]))
    print(classification_report(y_test, pred_dt))

    print("GBC prediction score: {}".format(scores["score_gbc"]))
    print(classification_report(y_test, pred_gbc))

    print("RF prediction score: {}".format(scores["score_rfc"]))
    print(classification_report(y_test, pred_rfc))
