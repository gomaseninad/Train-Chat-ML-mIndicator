# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 12:42:20 2018

@author: Jayant
"""

import cleaning
import numpy as np
import glob
import pandas as pd
import pickle
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer


with open('tokenizer/tokenize_b.pickle', 'rb') as handle:
    tokenize = pickle.load(handle)
modelFileLoad1 = open('models/model_b', 'rb')
modelFileLoad2= open('models/model_m1', 'rb')
encoder = LabelEncoder()
encoder.classes_ = np.load('labelencoder/encoder_m1.npy')

fit_model1 = pickle.load(modelFileLoad1)
fit_model2 = pickle.load(modelFileLoad2)
from IPython.display import display
from tabulate import tabulate


def analyze_message(value):
    col_names = ['Station name','Train name','Category','Platform number','Is spam','If delay']
    output = pd.DataFrame(columns=col_names)
    #print(value)
    a,b,d,t = cleaning.clean(value)
    c, e, f = np.nan, np.nan, np.nan
    tex = pd.Series(value)
    numtext = tokenize.texts_to_matrix(tex)
    
    ee = fit_model1.predict(numtext)
    index = np.argmax(ee)
    if index == 0:
        e = False
    else:
        e = True
    if t is not np.nan:
        predicted = fit_model2.predict(t)
        index = np.argmax(predicted)
        c = encoder.inverse_transform([index])
        f = cleaning.get_delay_time(c, value)
        #print(encoder.inverse_transform([index]))
    output = output.append(pd.Series([a, b, c, d, e, f], index=col_names ), ignore_index=True)
    print(tabulate(output, headers=col_names, tablefmt='psql'))   
    
print("Enter message: ")
text = input()
analyze_message(text)