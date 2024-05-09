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
n_values = list(range(0, 200, 10))
n_times = [None]*len(n_values)

with PixelBar('Performing requests...', max=np.sum(n_values)) as bar:
    for idx_Nv, set_N in enumerate(n_values):
        start_time = time.time()

        for idx, row in df_test.sample(n=set_N).iterrows():
            r = requests.post("http://127.0.0.1:5000/classify", json={"text": row["text"]})
            results = r.json()

            bar.next()

        n_times[idx_Nv] = (time.time() - start_time)

# Plot graph of time vs req vals
plt.plot(n_values, n_times)
plt.xlabel('Number of requests')
plt.ylabel('Response time (s)')
plt.show()