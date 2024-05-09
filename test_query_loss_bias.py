import requests, json
import pandas as pd
from progress.bar import *
from progress.spinner import *
from sklearn.utils import shuffle

df_test = pd.read_csv("data/orig/WELFake_Dataset.csv")
df_test.drop_duplicates(inplace = True)
df_test.dropna(inplace = True)
#df_test['text'] = df_test['text'] + " " + df_test['title'] #80.76% with this, same without
df_test.drop(['title'], axis=1, inplace=True)
df_test = shuffle(df_test)

# Stat values
set_N = 70000
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

print("Percent predicted correct: ", counter_stats["pass"]/set_N*100, "%")

print("""Test statistics (n={})
- passed {}
- failed {}
- loss {}%

{}""".format(set_N, counter_stats["pass"], counter_stats["fail"], loss_dev, "Tends to estimate articles as " + ("true" if loss_dev < 0 else "false")))