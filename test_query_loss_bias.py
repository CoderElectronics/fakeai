import requests, json
import pandas as pd
from progress.bar import *
from progress.spinner import *

def df_preproc(dfm):
    dfm = dfm.drop(["title", ], axis=1)
    dfm = dfm.loc[:, ~dfm.columns.str.contains('^Unnamed')]
    dfm = dfm.dropna(subset=['text'])
    dfm = dfm[dfm["text"].str.strip() != ""]
    return dfm

df_test = df_preproc(pd.read_csv("data/dev_nobert.csv"))

# Stat values
set_N = 5
counter_stats = {
    "pass": 0,
    "fail": 0,
    "sum": 0,
}

with PixelBar('Loading articles...', max=set_N) as bar:
    for idx, row in df_test.sample(n=set_N).iterrows():
        r = requests.post("http://127.0.0.1:5000/classify", json={"text": row["text"]})
        results = r.json()
        #print("Declared Truth: {}\nReal Results: \n{}\n".format(row["label"], json.dumps(results, indent=2)))

        if row["label"] - results["weighted_avg"] > 0.5 or row["label"] - results["weighted_avg"] < -0.5:
            counter_stats["fail"] += 1
        else:
            counter_stats["pass"] += 1

        counter_stats["sum"] += (row["label"]-results["weighted_avg"])
        bar.next()

loss_dev = (counter_stats["sum"]/set_N)*100

print("""
Test statistics (n={})
- passed {}
- failed {}
- loss {}%

{}""".format(set_N, counter_stats["pass"], counter_stats["fail"], loss_dev, "Tends to estimate articles as " + ("true" if loss_dev < 0 else "false")))