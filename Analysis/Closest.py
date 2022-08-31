"""Find the closest sentence to a generated sentence"""
from scipy.spatial.distance import cdist
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
sentence="he ' s is a career-defining ."
dataset="SST-2"
# algo="VAE"
dataset_size="100"

df=list(pd.read_csv(f"../SelectedData/{dataset}/{dataset_size}/train.tsv", sep='\t')['sentence'])
print(df)

count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

training=df+[sentence]
count_vect=count_vect.fit(training)
tf=tfidf_transformer.fit(count_vect.transform(training))

df_count=tf.transform(count_vect.transform(df)).todense()
sentence_count=tf.transform(count_vect.transform([sentence])).todense()

k=10

dist=np.array(cdist(sentence_count, df_count, 'cosine'))
max=np.argpartition(dist, k).reshape(-1)[:k]
for index in max:
    print(df[index])