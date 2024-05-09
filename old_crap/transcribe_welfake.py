import pandas as pd
from sklearn.model_selection import train_test_split
from progress.bar import *

ds_welfake = pd.read_csv("data/orig/WELFake_Dataset.csv")

# target frame
frame = {
    "idxs": [],
    "labels": [],
    "texts": []
}

ir_idx = 0
with PixelBar('Transcribing WELFake Contents...', max=ds_welfake.shape[0]) as bar:
    for index, row in ds_welfake[ds_welfake.text.notnull()].iterrows():
        frame['idxs'].append(ir_idx)
        frame['labels'].append(row["label"])
        frame['texts'].append(row["text"] + (row["title"] if isinstance(row["title"], str) else ''))

        bar.next()
        ir_idx += 1

df = pd.DataFrame({
    'id': frame["idxs"],
    'label': frame["labels"],
    'alpha': ['a']*len(frame["idxs"]),
    'text': frame["texts"]
})

df_bert_gr1, df_bert_test = train_test_split(df, test_size=0.25)
df_bert_train, df_bert_dev = train_test_split(df_bert_gr1, test_size=0.05)

df_bert_train.to_csv('data/train.tsv', sep='\t', index=False, header=False)
df_bert_dev.to_csv('data/dev.tsv', sep='\t', index=False, header=False)
df_bert_test.to_csv('data/test.tsv', sep='\t', index=False, header=False)

# Sep non-bert data
df_nobert_gr1, df_nobert_test = train_test_split(ds_welfake, test_size=0.25)
df_nobert_train, df_nobert_dev = train_test_split(df_nobert_gr1, test_size=0.05)

df_nobert_train.to_csv('data/train_nobert.csv')
df_nobert_dev.to_csv('data/dev_nobert.csv')
df_nobert_test.to_csv('data/test_nobert.csv')