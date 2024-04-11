import pandas as pd
import tarfile
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#load data
all_data = pd.read_csv('data/allData.tsv', sep='\t')

train_raw = all_data[all_data.text.not_null()]