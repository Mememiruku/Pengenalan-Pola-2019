import twitter
import pandas as pd
import numpy as np


auth = [line.strip('\n') for line in open('auth_key.txt', 'r')]
api = twitter.Api(  consumer_key        =auth[0],
                    consumer_secret     =auth[1],
                    access_token_key    =auth[2],
                    access_token_secret =auth[3],
                    cache=None,
                    tweet_mode= 'extended')

    
tweet = []
for result in api.GetSearch(term=('Tolol'), since=2019-2-17, count=100):
    tweet.append([result.user.id, result.user.name, result.created_at, result.full_text])

tweet2 = []
for result in api.GetSearch(term=('tolol'), since=2015-2-17, count=100):
    tweet2.append([result.user.id, result.user.name, result.created_at, result.full_text])

out = pd.DataFrame(tweet, columns=['user_id', 'username', 'date', 'Tweet'])
out.to_csv("output.csv")

out2= pd.DataFrame(tweet2, columns=['user_id', 'username', 'date', 'Tweet'])
out2.to_csv("output2.csv")