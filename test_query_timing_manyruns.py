import requests, json, time
import pandas as pd
import numpy as np
from progress.bar import *
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

df_test = pd.read_csv("data/orig/WELFake_Dataset.csv")
df_test.drop_duplicates(inplace = True)
df_test.dropna(inplace = True)
#df_test['text'] = df_test['text'] + " " + df_test['title'] #80.76% with this, same without
df_test.drop(['title'], axis=1, inplace=True)
df_test = shuffle(df_test)

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