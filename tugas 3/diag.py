import pandas as pd
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
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
tokenizer = nltk.RegexpTokenizer(r'\w+')

for index, 

#Bagi file menjadi dua, data Training dan data Testing
train_data = df[df['diagnosis'].notnull()]
test_data = df[df['diagnosis'].isnull()]
train_data.reset_index(inplace=True, drop=True)
train_data.head()
test_data.reset_index(inplace=True, drop=True)
test_data.head()


