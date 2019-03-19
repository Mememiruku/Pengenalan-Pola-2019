import pandas as pd
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from pathlib import Path
import math


#fetch file
path = Path(__file__).parents[0]
inputfile = str(path) + "\\Diagnosa.csv"
df = pd.read_csv(inputfile, names=["query","diagnosis"], header=0)


#preprocessing data
factory = StemmerFactory()
stemmer = factory.create_stemmer()

for index, row in df.iterrows():
    #stemming
    stem = stemmer.stem(row[0])
    #case folding
    words = stem.lower()
    df.at[index, "query"] = words

#Feature Extraction
vector = CountVectorizer()
transform = np.array(vector.fit_transform(df['query']).toarray())
unique_words = len(transform[0])
for words in range(unique_words):
    df.insert(2+words, 'word_{}'.format(words+1), transform[:,words])

df.head()

#Bagi file menjadi dua, data Training dan data Testing
train_data = df[df['diagnosis'].notnull()]
test_data = df[df['diagnosis'].isnull()]
train_data.reset_index(inplace=True, drop=True)
train_data.head()
test_data.reset_index(inplace=True, drop=True)
test_data.head()

#Minkowski Distance
def minkowski_distance(x,y,p_value):
    return nth_root(sum(pow(abs(a-b), p_value) for a, b in zip(x, y)), p_value)

#calculate nth root
def nth_root(value, n_root):
    root_value = 1/float(n_root)
    return round(value ** root_value, 3)

#Calculate distance using Minkowski Distance Algorithm with power set to 3
test_data.insert(2, 'distance', 0.0)
for test_index, test_row in test_data.iterrows():
    distance = []
    test_vector = np.array(test_row[3:].tolist())
    for train_index, train_row in train_data.iterrows():
        train_vector = np.array(train_row[2:].tolist())
        distance += [minkowski_distance(test_vector, train_vector, 3)]
    test_data.at[test_index, 'distance'] = np.min(distance)
    test_data.at[test_index, 'diagnosis'] = train_data.at[np.argmin(distance), 'diagnosis']

#show test data distance and diagnose
print(test_data.iloc[:, :3])

#create csv output
test_data.to_csv('hasil.csv')

