#!/usr/bin/env python
# coding: utf-8

# In[9]:


import re
import json
import gensim
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from keras.models import model_from_json
from nltk.tokenize import WordPunctTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# In[2]:


def tweet_cleaner(text):
    tok = WordPunctTokenizer()
    x=text
    # hapus rt
    cl = re.sub(r'\s*RT\s*@[^:]*:.*', '', x)
    cl = re.sub(r'\s*rt\s*@[^:]*:.*', '', cl)
    # hapus mention
    cl = re.sub(r'@[A-Za-z0-9]([^:\s]+)+', '', cl)
    # hapus link
    cl = re.sub(r'https?://[A-Za-z0-9./]+', '', cl)
    # hapus hashtag
    cl = re.sub(r'(?:\s|^)#[A-Za-z0-9\-\.\_]+(?:\s|$)', '', cl)
    # kata ulang
    cl = re.sub(r'\w*\d\w*', '', cl)
    cl = re.sub(r'\b(\w+)(\1\b)+', r'\1', cl)
    # hapus simbol
    cl = re.sub(r'[^a-zA-Z]', ' ', cl)
    # lower
    cl=cl.lower()
    # format teks 
    cl=tok.tokenize(cl)
    cl=(" ".join(cl))
    return cl

def stopword(text):
    stopword_ = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    result = [i for i in tokens if not i in stopword_]
    result=' '.join(result)
    return result

def stem(text):
    stemmer= PorterStemmer()
    result=[]
    text = word_tokenize(text)
    for word in text:
        result.append(stemmer.stem(word))
    result=' '.join(result) 
    return result



def tokenn(input_clean):
    with open('lib/tokenizer_eng.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)                                 
    sequences = tokenizer.texts_to_sequences(input_clean)               
    len(tokenizer.word_index)
    #
    length = []
    for x in input_clean:
        length.append(len(x.split()))
    max(length)
    return sequences

# In[6]:


def pad(sequences):
    x_train_seq = pad_sequences(sequences, maxlen=70)
    return x_train_seq


# In[7]:

json_file = open('lib/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('lib/weights.hdf5')

global graph
graph = tf.get_default_graph()

def predict(text,clean):
    input_clean=text
    print(input_clean)
    
    if clean!=True:
        # text cleaning
        input_clean = tweet_cleaner(input_clean)
        print(input_clean)
        # stopwords
        input_clean = stopword(input_clean)
        print(input_clean)
        # stemming
        input_clean = stem(input_clean)
        print(input_clean)
    # simpan ke dataframe
    df=pd.DataFrame([input_clean], columns=['text'])
    input_clean=df.text
    # tokenizing
    sequences=tokenn(input_clean)
    # padding
    input_ready=pad(sequences)
    # predict classes
    with graph.as_default():
        prediction = loaded_model.predict_classes(input_ready).tolist()
    
    return json.dumps(prediction)


# In[10]:


#input= "@AmericanAir so bad can't take this anymore. never again"
#print(predict(input))


# In[ ]:


