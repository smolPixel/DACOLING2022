"""Gives all kind of stats on the datasets"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#
# path="/media/frederic/DAGlue/jiant-v1-legacy/data/OG"
#
# """SST-2"""
# task="SST-2"
# dfTrain=pd.read_csv(f"{path}/{task}/train.tsv", sep='\t')
# sent=dfTrain['sentence']
# sent=[len(se.split()) for se in sent]
# print(f"SST 2 max length of sentence : {max(sent)}")
#
# """CoLA"""
# task="CoLA"
# dfTrain=pd.read_csv(f'{path}/{task}/train.tsv', names=["", "label", "s", "sentence"], header=None, sep='\t')
# sent=dfTrain['sentence']
# sent=[len(se.split()) for se in sent]
# print(f"CoLA max length of sentence : {max(sent)}")

"""FakeNews"""
df=pd.read_csv(f'/media/frederic/DAGlue/data/FakeNews/train.csv', index_col=0).dropna()
df=df[['title', 'label']]
df=df.rename(columns={"title":"sentence"})
train, test= train_test_split(df, test_size=0.3)

# # dfTest=pd.read_csv(f'/media/frederic/DAGlue/data/FakeNews/dev.csv', index_col=0)
# train=df
# # test=dfTest[['title', 'label']].dropna()
# test.rename(columns={"title":"sentence"})
train.to_csv(f'/media/frederic/DAGlue/data/FakeNews/train.tsv', sep='\t', index=False)
test.to_csv(f'/media/frederic/DAGlue/data/FakeNews/dev.tsv', sep='\t', index=False)