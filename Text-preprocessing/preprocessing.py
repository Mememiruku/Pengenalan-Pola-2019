import pandas as pd
import numpy as numpy
import string

df = pd.read_csv('D:\\College lyfe\\Semester 6\\Pengenalan Pola\\Hate-speech-recognition\\output.csv', dtype={'Tweet':str})

#Case Folding
hasil=[]
for x in range (0, 100):
    input_str = df.at[x, 'Tweet']
    hasil.append([input_str.lower()])
new_df = pd.DataFrame(hasil, columns=['Tweet'])
df.update(new_df)

#Remove Punctuation
hasil2=[]
for x in range (0, 100):
    input_str = df.at[x, 'Tweet']
    hasil2.append(input_str.translate(str.maketrans("","", "!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~")))
new_df2 = pd.DataFrame(hasil2, columns=['Tweet'])
df.update(new_df2)

#Tokenization
for x in range (0, 100):
    input_str = df.at[x, 'Tweet']
    df.at[x, 'Tweet'] = input_str.split()

print(df)
df.to_csv("output.csv")