#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 20:38:07 2018

@author: ztw1e12
"""

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition 
# from textblob import TextBlob 


import pandas, numpy, string
from keras import layers, models, optimizers
from sklearn.linear_model import LogisticRegression


from sklearn.linear_model import Ridge
import random



# load the dataset

df = pandas.read_csv("sentiment labelled sentences/yelp_labelled.txt", names=['sentence', 'label'], sep="\t")


# Splitting the data (sentence and labbels) into training and testing data 

sentences = df['sentence'].values
y = df['label'].values


# create a dataframe using texts and lables
trainDF = pandas.DataFrame()
trainDF['text'] = sentences
trainDF['label'] = y



# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])



# Will be used to train models

trainDF['char_count'] = trainDF['text'].apply(len)
trainDF['word_count'] = trainDF['text'].apply(lambda x: len(x.split()))
trainDF['word_density'] = trainDF['char_count'] / (trainDF['word_count']+1)
trainDF['punctuation_count'] = trainDF['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
trainDF['title_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
trainDF['upper_case_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))


'''
pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}


# function to check and get the part of speech tag count of a words in a given sentence
def check_pos_tag(x, flag):
    cnt = 0
    try:
        wiki = TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt

trainDF['noun_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'noun'))
trainDF['verb_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'verb'))
trainDF['adj_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adj'))
trainDF['adv_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adv'))
trainDF['pron_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'pron'))
'''



# Our list of functions to apply.
transform_functions = [
        trainDF['char_count'],
        trainDF['word_count'],
        trainDF['word_density'],
        trainDF['punctuation_count'],
        trainDF['title_word_count'],
        trainDF['upper_case_word_count']
]

# Apply each function and put the results into a list.
columns = []
for func in transform_functions:
    columns.append(func)
    
# Convert the meta features to a numpy array.
meta = numpy.asarray(columns).T

print (meta)


train_rows = 750
# Set a seed to get the same "random" shuffle every time.
random.seed(1)

# Shuffle the indices for the matrix.
indices = list(range(meta.shape[0]))
random.shuffle(indices)

# Create train and test sets.
train = meta[indices[:train_rows], :]
test = meta[indices[train_rows:], :]
# train = numpy.nan_to_num(train)

# Run the regression and generate predictions for the test set.
reg = Ridge(alpha=.1)
reg.fit(train,train_y)
predictions = reg.predict(test)