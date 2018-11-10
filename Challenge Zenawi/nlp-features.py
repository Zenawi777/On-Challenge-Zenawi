#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 20:38:07 2018

@author: ztw1e12
"""
from sklearn import model_selection
# from textblob import TextBlob 
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
import pandas, numpy, string
from keras import layers
import random


# Load dataset using pandas library as sentence and labels
df = pandas.read_csv("sentiment labelled sentences/yelp_labelled.txt", names=['sentence', 'label'], sep="\t")


# Getting the the sentence and labels dataset from the loaded data
sentences = df['sentence'].values
y = df['label'].values


# Create a dataframe using texts and lables
trainDF = pandas.DataFrame()
trainDF['text'] = sentences
trainDF['label'] = y


# Splitting the data into training and testing (validation) data sets
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])


# NLP features for training models
trainDF['char_count'] = trainDF['text'].apply(len)
trainDF['word_count'] = trainDF['text'].apply(lambda x: len(x.split()))
trainDF['word_density'] = trainDF['char_count'] / (trainDF['word_count']+1)
trainDF['punctuation_count'] = trainDF['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
trainDF['title_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
trainDF['upper_case_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))


# List of functions to train model
transform_functions = [
        trainDF['char_count'],
        trainDF['word_count'],
        trainDF['word_density'],
        trainDF['punctuation_count'],
        trainDF['title_word_count'],
        trainDF['upper_case_word_count']
]


# Applying each function and put the result into a list.
columns = []
for func in transform_functions:
    columns.append(func)
  
    
# Convert the meta features to a numpy array and transposing the data.
meta = numpy.asarray(columns).T

print ('=====================================================================')

print (meta)

print ('=====================================================================')


# Set a seed to get the same "random" shuffle every time
train_rows = 750
random.seed(1)


# Shuffling matrix indices  
indices = list(range(meta.shape[0]))
random.shuffle(indices)


# Create train and test data sets after shuffle
train = meta[indices[:train_rows], :]
test = meta[indices[train_rows:], :]


# Tokenizing the data set
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_x)
X_train = tokenizer.texts_to_sequences(train_x)
X_test = tokenizer.texts_to_sequences(valid_x)


# Building cnn model
vocab_size = len(tokenizer.word_index) + 1  # 1 is added due the reserved index 0 
embedding_dim = 6
maxlen = 6
model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
model.add(layers.Conv1D(130, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Training and testing the model
profile = model.fit(train, train_y, epochs=15, verbose=True, validation_data=(test, valid_y), batch_size=15)

# Evaluating model
loss, accuracy = model.evaluate(train, train_y, verbose=True)
loss, accuracy = model.evaluate(test, valid_y, verbose=True)