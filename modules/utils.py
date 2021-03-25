import re
import string
import sqlite3

import numpy as np
import pandas as pd

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from wordcloud import WordCloud
import string


import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns; sns.set()

from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, f1_score, recall_score, confusion_matrix


#FUNCTIONS USED FOR DATA COLLECTION
def target_mod (text):
    if text == '1 star rating' or text == '2 star rating':
        return 'Negative'
    
    if text == '4 star rating' or text == '5 star rating':
        return 'Positive'
    
    else: return 'Neutral'
    
    
def target_to_numeric(text):
    
    if text == 'Negative':
        return 0
    
    if text == 'Positive':
        return 5
    
    else: return 3
    

#FUNCTIONS USED FOR DATA PROCESSING AND CLEANING

# create a function for cleaning text of each review with regex
def cleaning_text(text):
    '''
    Looking for speciffic patterns in the text and 
    removing them or replacing with space
    Function returns string
    ''' 
    # make text lowercase
    text = text.lower()
    
    # string punctuations
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    
    # removing patterns and replace it with nothing
    text = re.sub('\[.*?\]', '', text)
 
    # removing digits if they surounded with text or digit
    text = re.sub('\w*\d\w*', '', text)
    
    # make just 1 space if there is more then 1
    text = re.sub('\s+', ' ', text)
    
    # replace new line symbol with space
    text = re.sub('\n', ' ', text)
    
    # removing any quotes
    text = re.sub('\"+', '', text)
    
    # removing `rt`
    #text = re.sub('(rt)', '', text)

    return text


def filter_review(text, n = 2):
    """
    Filtering words in review by 
    number of letters
    n : number of letters(symbols) in the word
    """
    return  ' '.join([word for word in text.split(' ') if len(word) > n])


def tokenize_text(text):
    
    """
    Tocanize text 
    Wiil filter data with stopwords
    """
    stopwords_list = stopwords.words('english')
    stopwords_list += list(string.punctuation)
    tokens = nltk.word_tokenize(text)
    
    stopwords_removed = [token for token in tokens if token not in stopwords_list] 
    
    return stopwords_removed

# function to creat a list with all lemmatized words

def lematizing_text(data):
    
    """
    Lematizing words from the corpus data
    Returns list of strings with lematized 
    words in each string
    """
    
    lemmatizer = WordNetLemmatizer() 
    lemmatized_output = []

    for tweet in data:
        lemmed = ' '.join([lemmatizer.lemmatize(w) for w in tweet])
        lemmatized_output.append(lemmed)
        
    return lemmatized_output


# FUNCTIONS USED FOR MODELING


def modifier (text):
    if '1' in text: return 1
    if '2' in text: return 2
    if '3' in text: return 3
    if '4' in text: return 4
    if '5' in text: return 5
    

def metrics(train_preds, y_train, test_preds, y_test):
    
    print(f"Training Accuracy:\t{accuracy_score(y_train, train_preds):.4}",
          f"\tTesting Accuracy:\t{accuracy_score(y_test, test_preds):.4}")

    print(f"Training Precision:\t{precision_score(y_train, train_preds, average='weighted'):.4}",
          f"\tTesting Precision:\t{precision_score(y_test, test_preds, average='weighted'):.4}")

    print(f"Training Recall:\t{recall_score(y_train, train_preds, average='weighted'):.4}",
          f"\tTesting Recall:\t\t{recall_score(y_test, test_preds, average='weighted'):.4}")

    print(f"Training F1:\t\t{f1_score(y_train, train_preds, average='weighted'):.4}",
          f"\tTesting F1:\t\t{f1_score(y_test, test_preds, average='weighted'):.4}")
    