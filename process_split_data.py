from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import re, string
from string import punctuation
from progress.bar import *

import nltk

with PixelBar('', max=4) as bar:
    bar.bar_prefix = 'Loading NLTK Dataset...'
    bar.update()

    # Loading NLTK sets
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4') #If these are already downloaded, it will tell you in red text, do not be alarmed

    from nltk.corpus import stopwords
    from textblob import Word

    bar.next()
    bar.bar_prefix = 'Loading original datasets...'
    bar.update()

    # Loading dataset files
    true_news = pd.read_csv("data/orig/Fake.csv")
    fake_news = pd.read_csv("data/orig/True.csv")

    true_news['news_class'], fake_news['news_class'] = 1, 0
    news = pd.concat([true_news, fake_news])
    news.drop_duplicates(inplace = True)

    bar.next()
    bar.bar_prefix = 'Preprocessing data...'
    bar.update()

    # Preprocessing
    news['text'] = news['text'] + " " + news['title'] #add title to text
    news.drop(['title', 'date', 'subject' ], axis =1, inplace=True ) #drop the title, date, and subject
    news.rename(columns={'news_class': 'label'}, inplace=True)

    news['text'] = news['text'].apply(lambda x : " ".join(x.lower() for x in x.split() ) )
    news['text'] = news['text'].str.replace('[^\w\s]','')
    news['text'] = news['text'].str.replace('\d', '' )

    stop_words = set(stopwords.words('english')) #get english stopwords
    punctuation = list(string.punctuation) #get punc
    stop_words.update(punctuation)
    news['text'] = news['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words )) #remove stopwords and punc
    news['text'] = news['text'].apply(lambda x : " ".join([Word(word).lemmatize() for word in x.split()]) )
    news['text'] = news['text'].apply(lambda x : " ".join(re.sub(r'http\S+', '', x ) for x in x.split() ) )

    bar.next()
    bar.bar_prefix = 'Exporting data...'
    bar.update()

    # Splitting and exporting
    train, test = train_test_split(news, test_size=0.25, shuffle=True, random_state=11 )

    train.to_csv('data/train.csv', index=False)
    test.to_csv('data/test.csv', index=False)

    bar.next()