# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 03:15:19 2018

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

# =============================================================================
# list1 = ['train late by 20 min (sent from Ambernath Stn.)',
#          '4:48 KALYAN SLOW TRAIN IS CANCELLED FOR TODAY  (sent from Cst Stn.)',
#          
#          'kal kurla to thane 10 am ko milegi kya? (sent from Mankhurd)'
#          ,
#       
#       '6:26 Cst  slow train shortly arriving on platform no 3 (sent from Dadar Stn.)'
#       ,'Asangaon fast train halted on platform no 7 (sent from Cst Stn.)','hi',
#       'dadar cst fast train is late by 20 minutes']
# # Loading logistic regression model from training set 1  
#   
# =============================================================================
path =r'Hackathon ML data'
allFiles = glob.glob(path + "/*.txt")

list_files = []
count = 0

for file_ in allFiles:
    if count < 60:
        count +=1
        continue
    if count == 62:
        break
    if count%5 == 0:
        print("Reading %d files..."%count)
    df = pd.read_json(file_)
    list_files.append(df)
    count += 1

train = pd.concat(list_files, axis = 1, ignore_index = True)
train = train.T;
with open('tokenizer/tokenize_b.pickle', 'rb') as handle:
    tokenize = pickle.load(handle)
modelFileLoad1 = open('models/model_b', 'rb')
modelFileLoad2= open('models/model_m1', 'rb')
encoder = LabelEncoder()
encoder.classes_ = np.load('labelencoder/encoder_m1.npy')

fit_model1 = pickle.load(modelFileLoad1)
fit_model2 = pickle.load(modelFileLoad2)

col_names = ['Message','Station name','Train name','Category','Platform number','Is spam','If delay']
output = pd.DataFrame(columns=col_names)

for idx, value in enumerate(train.m):
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
    else:
        pass
    output = output.append(pd.Series([value, a, b, c, d, e, f], index=output.columns ), ignore_index=True)
