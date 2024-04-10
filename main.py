import pandas as pd

#load data
all_data = pd.read_csv('data/allData.tsv', sep='\t')
train = all_data[:Y]
dev = all_data[Y:Z] #TODO: find X, Y, Z. Train is large, dev is 1% of train, test is ~20%?
test = all_data[Z:]