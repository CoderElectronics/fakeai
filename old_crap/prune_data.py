import pandas as pd
import pagecurl as pc
import progressbar as pb
import numpy as np

ds_fnn = pd.read_csv("../data/FakeNewsNet.csv")
ds_welfake = pd.read_csv("../data/WELFake_Dataset.csv")

rowN = ds_fnn.shape[0] + ds_welfake.shape[0]

# target frame
target_data = [[]*rowN, []*rowN, []*rowN, []*rowN]

final_idx = 0
for index, row in ds_fnn.iterrows():
    pb.pbar(index, ds_fnn.shape[0], prefix='Downloading FNN Site Contents:', suffix='Complete', length=50)

    print(index)
    df.iloc[index]['id'] = index
    df.iloc[index]['label'] = row["real"]
    df.iloc[index]['text'] = pc.get_content(row['news_url']) + row["title"]
    print(pc.get_content(row['news_url']))
    final_idx += 1

for index, row in ds_welfake.iterrows():
    pb.pbar(index, ds_welfake.shape[0], prefix='Downloading WELFake Site Contents:', suffix='Complete',
                     length=50)

    df.iloc[final_idx+index]['id'] = final_idx+index
    df.iloc[final_idx+index]['label'] = row["label"]
    df.iloc[final_idx+index]['text'] = row['text'] + row['title']

df.to_csv('data/target_bert.tsv', sep="\t")

df = pd.DataFrame({
    'id': [np.nan]*rowN,
    'label': ['']*rowN,
    'alpha': ['a']*rowN,
    'text': ['']*rowN
})
