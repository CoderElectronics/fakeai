import re, string, json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump, load
from pathlib import Path
from datetime import datetime

from progress.bar import *
from progress.spinner import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Load data sets
Path("data/models").mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")

#split data
with PixelBar('', max=6) as bar:
    bar.bar_prefix = 'Preprocessing data...'
    bar.update()

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
    LR.fit(xv_train, y_train)
    dump(LR, 'data/models/lr.joblib')
    pred_lr = LR.predict(xv_test)

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
        "score_rfc": RFC.score(xv_test, y_test),
        "num_training_articles": df.shape[0],
        "model_date": datetime.today().strftime('%Y-%m-%d')
    }
    json.dump(scores, open('data/models/score_weights.json', mode='w'))

    # Results
    fig, axs = plt.subplots(figsize=(10, 5), nrows=2, ncols=2)

    ConfusionMatrixDisplay.from_predictions(y_test, pred_lr, ax=axs[0, 0])
    axs[0, 0].xaxis.set_ticklabels(["true", "fake"])
    axs[0, 0].yaxis.set_ticklabels(["true", "fake"])
    _ = axs[0, 0].set_title(
        f"Confusion Matrix for Logistic Regression"
    )

    ConfusionMatrixDisplay.from_predictions(y_test, pred_dt, ax=axs[1, 0])
    axs[1, 0].xaxis.set_ticklabels(["true", "fake"])
    axs[1, 0].yaxis.set_ticklabels(["true", "fake"])
    _ = axs[1, 0].set_title(
        f"Confusion Matrix for Decision Tree Classifier"
    )

    ConfusionMatrixDisplay.from_predictions(y_test, pred_gbc, ax=axs[0, 1])
    axs[0, 1].xaxis.set_ticklabels(["true", "fake"])
    axs[0, 1].yaxis.set_ticklabels(["true", "fake"])
    _ = axs[0, 1].set_title(
        f"Confusion Matrix for Gradient Boosting Classifier"
    )

    ConfusionMatrixDisplay.from_predictions(y_test, pred_rfc, ax=axs[1, 1])
    axs[1, 1].xaxis.set_ticklabels(["fake", "true"])
    axs[1, 1].yaxis.set_ticklabels(["fake", "true"])
    _ = axs[1, 1].set_title(
        f"Confusion Matrix for Random Forest Classifier"
    )

    print("\n\nNumber of training articles: {}\n".format(scores["num_training_articles"]))

    print("Logistic regression prediction score: {}\n".format(scores["score_gbc"]))

    print("DTC prediction score: {}".format(scores["score_dt"]))
    print(classification_report(y_test, pred_dt))

    print("GBC prediction score: {}".format(scores["score_gbc"]))
    print(classification_report(y_test, pred_gbc))

    print("RF prediction score: {}".format(scores["score_rfc"]))
    print(classification_report(y_test, pred_rfc), end="")

    plt.show()
