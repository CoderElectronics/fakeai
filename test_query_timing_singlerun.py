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
set_N = 200
time_N = [None]*set_N

with PixelBar('Performing requests...', max=set_N) as bar:
    idx_req = 0

    for idx, row in df_test.sample(n=set_N).iterrows():
        start_time = time.time()

        r = requests.post("http://127.0.0.1:8003/classify", json={"text": row["text"]})
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