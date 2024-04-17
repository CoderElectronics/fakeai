import requests, json, time
import pandas as pd
import numpy as np
from progress.bar import *
import matplotlib.pyplot as plt

def df_preproc(dfm):
    dfm = dfm.drop(["title", ], axis=1)
    dfm = dfm.loc[:, ~dfm.columns.str.contains('^Unnamed')]
    dfm = dfm.dropna(subset=['text'])
    dfm = dfm[dfm["text"].str.strip() != ""]
    return dfm

df_test = df_preproc(pd.read_csv("data/dev_nobert.csv"))

# Stat values
set_N = 200
time_N = [None]*set_N

with PixelBar('Performing requests...', max=set_N) as bar:
    idx_req = 0

    for idx, row in df_test.sample(n=set_N).iterrows():
        start_time = time.time()

        r = requests.post("http://127.0.0.1:5000/classify", json={"text": row["text"]})
        results = r.json()

        time_N[idx_req] = (time.time() - start_time)
        idx_req += 1

        bar.next()

print("\nMax response time of {} at {}s".format(np.max(time_N), np.argmax(time_N)))

# Plot graph of time vs req vals
plt.plot(range(set_N), time_N)
plt.xlabel('Request # (idx/set_N)')
plt.ylabel('Response time (s)')
plt.show()