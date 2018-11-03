#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 11:40:03 2018

@author: ztw1e12
"""

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression


df = pd.read_csv("sentiment labelled sentences/yelp_labelled.txt", names=['sentence', 'label'], sep="\t")


# Splitting the data (sentence and labbels) into training and testing data 

sentences = df['sentence'].values
y = df['label'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

print (sentences_train.shape)


# vectorizing and tokonizing text; this is followed by fitting the data
vectorizer = TfidfVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)

print (X_train.shape)


# Training model LogisticRegression using data tokenized with CountVector 

classifier = LogisticRegression()
tst=classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)

print("Test Accuracy:",score)