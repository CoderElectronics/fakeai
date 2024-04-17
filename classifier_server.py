import re, string, json
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

# Load models
model_LR = load('data/models/lr.joblib')
model_DT = load('data/models/dt.joblib')
model_GBC = load('data/models/gbc.joblib')
model_RFC = load('data/models/rfc.joblib')

score_weights = json.load(open('data/models/score_weights.json'))

def df_preproc(dfm):
    dfm = dfm.drop(["title", ], axis=1)
    dfm = dfm.loc[:, ~dfm.columns.str.contains('^Unnamed')]
    dfm = dfm.dropna(subset=['text'])
    dfm = dfm[dfm["text"].str.strip() != ""]
    return dfm

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

df = pd.read_csv("data/train_nobert.csv")
df = df_preproc(df)
df["text"] = df["text"].apply(wordopt)

x_train, y_train = df["text"], df["label"]

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)

def query_ds(news):
    testing_news = {"text": [news]}

    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)

    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)

    pred_LR = model_LR.predict(new_xv_test)[0]
    pred_DT = model_DT.predict(new_xv_test)[0]
    pred_GBC = model_GBC.predict(new_xv_test)[0]
    pred_RFC = model_RFC.predict(new_xv_test)[0]

    return {
        "pred_lr": int(pred_LR),
        "pred_dt": int(pred_DT),
        "pred_gbc": int(pred_GBC),
        "pred_rfc": int(pred_RFC),
        "weighted_avg": ((pred_LR*score_weights["score_lr"])
                         + (pred_DT*score_weights["score_dt"])
                         + (pred_GBC*score_weights["score_gbc"])
                         + (pred_RFC*score_weights["score_rfc"]))
                        / (score_weights["score_lr"]
                           + score_weights["score_dt"]
                           + score_weights["score_gbc"]
                           + score_weights["score_rfc"])
    }

# Server
app = Flask(__name__)

@app.route("/")
def root():
    return "<p>Send a query to /classify to test the model.</p>"

@app.route("/classify", methods=["POST"])
def query_classify():
    print("req'd")
    if (request.method == 'POST'):
        news = request.json.get('text')
        print("Dataset query sent: \n{}...(shortened)".format(news[0:100]))

        return jsonify(query_ds(news))

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)