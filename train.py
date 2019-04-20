# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 12:21:50 2018

@author: Jayant
"""
import glob
import numpy as np
from nltk.tokenize import word_tokenize
import pandas as pd
from collections import Counter
import pickle
from time import time, strftime, gmtime
t0 = time()
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils
# =============================================================================
# train = pd.read_json('Hackathon ML data/20180701_CENTRAL_LINE.txt');
# train = train.T;
# print(train.shape)
# =============================================================================
t0 = time()
path =r'Hackathon ML data'
allFiles = glob.glob(path + "/*.txt")

list_files = []
count = 0

for file_ in allFiles:
    if count == 60:
        break
    if count%5 == 0:
        print("Reading %d files..."%count)
    df = pd.read_json(file_)
    list_files.append(df)
    count += 1

print ("File reading time " + str(strftime("%H:%M:%S",gmtime(time()-t0)))+"s")
print("Read all files")
train = pd.concat(list_files, axis = 1, ignore_index = True)
train = train.T;

print(train.shape)
train.drop_duplicates(subset=['i','m'],keep='first',inplace=True)
train = train.iloc[:,[1,2,4,7,9]] 
print(train.shape)

del path, allFiles, count, df, list_files, file_

# =============================================================================
# messages = []
# for _, row in train.iterrows():
#     messages.append(row['m'])
# 
# messages_words = [word_tokenize(i) for i in messages]
# 
# list_messages = []
# for line in messages_words:
#     list_messages.extend(line)
# 
# 
# counts = Counter(list_messages)
# print(counts)
# 
# =============================================================================
cleaned = train.copy()
cleaned.m = cleaned['m'].str.lower()
print(cleaned.shape)

import re
def spam(msg,ri):
    if(len(msg.split(' '))<=2):
        if(ri=='' or ri==np.nan ):
            return True
    match=re.search(r"([(\d)(\:)(\s)(\w)(\?)(\.)(\@)(\,)(\')(\!)(\&)(\")(\)(\%)*-]+) \(sent from",msg)
    #match=re.search(r"[A-Za-z0-9 _.,!"'/$]*"
    if(match and len(match.group(1).split(' '))<=2):
        if(ri=='' or ri==np.nan ):
            return True
    return False

train['spam']=train.apply(lambda row: spam(row['m'],row['ri']), axis=1)

# =============================================================================
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(train['m'],train['spam'],test_size=0.2,random_state=42)
# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer(analyzer = "word",   
#                              tokenizer = None,    
#                              preprocessor = None, 
#                              stop_words = None,   
#                              max_features = 10000) 
# vectorizer.fit(x_train)
# with open('vectorizer/vectorize1.pickle', 'wb') as handle:
#     pickle.dump(vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
# print(x_test.shape)
# x_train_trans=vectorizer.transform(x_train).toarray()
# x_test_trans = vectorizer.transform(x_test).toarray()
# #train_data_features[0]
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# #model = RandomForestClassifier(n_estimators = 100) 
# model = SVC()
# model = model.fit(x_train_trans,y_train.values.ravel()) 
# 
# y_pred=model.predict(x_test_trans)
# from sklearn.metrics import accuracy_score
# print(accuracy_score(y_test,y_pred))
# 
# 
# 
# =============================================================================


from sklearn.model_selection import train_test_split
train_messages, test_messages, train_spam, test_spam = train_test_split(train['m'], train['spam'], test_size=0.2, random_state=47)

max_words = 10000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(train_messages)
with open('tokenizer/tokenize_b1.pickle', 'wb') as handle:
    pickle.dump(tokenize, handle, protocol=pickle.HIGHEST_PROTOCOL)


x_train = tokenize.texts_to_matrix(train_messages)
x_test = tokenize.texts_to_matrix(test_messages)

y_train = (train_spam)
y_test = (test_spam)

num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

batch_size = 32
epochs = 2

# Build the model
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print("Fitting the model...")              
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)

print("Evalutating...")
score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test accuracy:', score[1])


modelFileSave = open('models/model_b1', 'wb')
pickle.dump(model, modelFileSave)
modelFileSave.close()  
 

tester = "5:37 kalyan fast train arrived on platform no 4 (sent from dadar stn.)"
whlist = ['what', 'where','when','kab','\?','kaha','status','kya','kuthe','kute','hi','please','plz','status','update','why','ky','ahe ka','how']
# =============================================================================
# station_list = ['Thane','Kalyan','thane','cst','central','Dadar','kalyan','Dombivli','Badlapur','Ghatkopar','CST','Kurla','Cst','Kalva','kurla','Mulund','Nagar','CSMT','badlapur','dadar','Vikhroli','Ambernath','karjat','Sion','ghatkopar','Vithalwadi','Parel','Bhandup','Mumbra','Ulhas','Thakurli','Shahad','Kanjur','Vidyavihar','Marg','csmt','kasara','sion','Nahur','Titwala','Churchgate','dombivli','Elphinstone','Diva','Kopar','ambernath','titwala','Airoli','Byculla','mumbra','Chunabhatti','mulund','Karjat','dombivali','Andheri','Matunga','kalwa','cstm','Masjid','Sandhurst','Tilak','Koparkhairne','bhandup','vikhroli','Chinchpokli','Dombivali','Bandra','Currey Road','Vidyavihar','Vikhroli','Diva','Ulhas']
# station_list = [item.lower() for item in station_list]
# from collections import OrderedDict
# station_list = OrderedDict.fromkeys(station_list)
# station_list = station_list.keys()
# =============================================================================
print("Removing what where...")
station_list = ['thane', 'kalyan', 'cst', 'dadar', 'dombivli', 'badlapur', 'ghatkopar', 'kurla', 'kalva', 'mulund', 'csmt', 'vikhroli', 'ambernath', 'karjat', 'sion', 'vithalwadi', 'parel', 'bhandup', 'mumbra', 'ulhas', 'thakurli', 'shahad', 'kanjur', 'vidyavihar', 'kasara', 'nahur', 'titwala', 'churchgate', 'elphinstone', 'diva', 'kopar', 'airoli', 'byculla', 'chunabhatti', 'dombivali', 'andheri', 'matunga', 'kalwa', 'cstm', 'masjid', 'sandhurst', 'tilak', 'koparkhairne', 'chinchpokli', 'bandra', 'currey road']
new_stations = ['csmt','masjid','sandhurst road','byculla','chinchpokli', 'curry road', 'parel', 'dadar','matunga','sion','kurla','vidyavihar','ghatkopar','vikhroli','kanjur marg','bhandup','nahur','mulund','thane','kalva','mumbra','diva jn','kopar','dombivli','thakurli','kalyan','vithalwadi','ulhas nagar','ambernath','badlapur','vangani','shelu','neral','bhivpuri road','karjat','palasdhari','kelavli','dolavli','lowjee','khopoli','shahad','ambivli','titwala','khadavli','vasind','asangaon','atgaon','khardi','kasara']
station_list.extend(new_stations)
station_list = set(station_list)
cleaned = cleaned[~cleaned.m.str.contains('|'.join(whlist))]
print(cleaned.shape)

print("Removing stating with...")
questions = ('is','are','any','did')
cleaned = cleaned[~cleaned.m.str.startswith(questions)]
print(cleaned.shape)
#print(len(station_list))

del tester, whlist, new_stations, questions

print("Removing stations starts...")
t1 = time()
# =============================================================================
# for index, row in cleaned.iterrows():
#     splitted = word_tokenize(row.m)
# # =============================================================================
# #     count = 0
# #     for element in splitted:
# #         if element in station_list:
# #             count += 1
# # =============================================================================
#     count = len([i for i in splitted if i in station_list])
#     if count < 2:
#         cleaned.drop(index, inplace=True)
# 
# print(cleaned.shape)   
# 
# =============================================================================
train_type = ['slow', 'fast']
def remove_stations(index, text):
    splitted = word_tokenize(text)
    #count = len([i for i in splitted if i in station_list])
    count = 0
    length = len(splitted)
    for idx, value in enumerate(splitted):
        if value in station_list:
            #print(idx, " ", value)
            count += 1
            if idx+1 < length:
                if splitted[idx+1] in train_type:
                    #print(value + " " + splitted[idx+1] + " train")
                    cleaned.at[index,'train-name'] = value + " " + splitted[idx+1] + " train"
            
    if count < 2:
        cleaned.drop(index, inplace=True)


#print(remove_stations(0, ' 6:26 Cst  slow train shortly arriving on platform no 3 (sent from Dadar Stn.)'))

cleaned.apply(lambda row: remove_stations(row.name,row['m']), axis=1)
print ("Text Cleaning time " + str(strftime("%H:%M:%S",gmtime(time()-t1)))+"s")

# =============================================================================
# def checker(str):
#         splitted = word_tokenize(str.lower())
#         print(splitted)
#         count = len([i for i in splitted if i in station_list])
#         if count < 2:
#             print("OUT")
#         else:
#             print("NOT OUT")
#             
# checker(tester)
# =============================================================================


print("Categorising data...")
import re     
cleaned['sentfrom']='default value'
def get_sent_from(text):
    match = re.search(r'sent from ([\w\.-]+)', text)
    if match:
        return match.group(1) # The username (group 1) 
    else:
        return np.nan
    
cleaned['sentfrom']=cleaned['m'].apply(get_sent_from)




def make_categories(text):
    
    match1 = re.search(r'cancel', text)
    if match1:
        return "cancelled"
    
    
    match1 = re.search(r'stuck', text)
    match2 = re.search(r'broke', text)
    match3 = re.search(r'halt', text)
    match4 = re.search(r'still', text)
    if match1 or match2 or match3 or match4:
        return "stuck"
    
    match1 = re.search(r'delay', text)
    match2 = re.search(r'late', text)
    if match1 or match2:
        return "delay"
    
    match1 = re.search(r'arriv', text)
    match2 = re.search(r'reach', text)
    if match1 or match2:
        return "arrived"
    
    match1 = re.search(r'start', text)
    match2 = re.search(r'depart', text)
    match3 = re.search(r'leav', text)
    match4 = re.search(r'left', text)
    if match1 or match2 or match3 or match4:
        return "depart"
    
    return np.nan

cleaned['category']=cleaned['m'].apply(make_categories)
print(cleaned['category'].value_counts())

def get_delay_time(cat, m):
    if cat == "delay":
        text=m
        match = re.search(r'([\d]+) min', text)
        if(match):
            return match.group(1)
        match = re.search(r'([\d]+)min', text)
        if(match):
            return match.group(1)
        return np.nan
    return np.nan
cleaned['delay_time'] = cleaned.apply(lambda row: get_delay_time(row['category'],row['m']), axis=1)


final_train = train.copy()

train.describe()
cleaned.category = cleaned.category.fillna("")
new_df = cleaned[['time','category','sentfrom','delay_time']].copy()
final_train = pd.merge(train, new_df, how='left', on='time')


del new_df
# =============================================================================
# match = re.search(r'left', "5:43 titwala fast train lefted from platform no 4 (sent from dadar stn.)")
# if match:
#     print('hi') # The username (group 1) 
# else:
#     print('bye')
# 
# =============================================================================

# Modeling
print("Modelling starts...")


df = cleaned.copy()
from sklearn.model_selection import train_test_split
train_messages, test_messages, train_category, test_category = train_test_split(df['m'], df['category'], test_size=0.2, random_state=47)

max_words = 10000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(train_messages)
with open('tokenizer/tokenize_m1.pickle', 'wb') as handle:
    pickle.dump(tokenize, handle, protocol=pickle.HIGHEST_PROTOCOL)


x_train = tokenize.texts_to_matrix(train_messages)
x_test = tokenize.texts_to_matrix(test_messages)


encoder = LabelEncoder()
train_category.unique()
encoder.fit(train_category)
y_train = encoder.transform(train_category)
y_test = encoder.transform(test_category)
np.save('labelencoder/encoder_m1.npy', encoder.classes_)

num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

batch_size = 32
epochs = 3

# Build the model
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print("Fitting the model...")              
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)

print("Evalutating...")
score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test accuracy:', score[1])

# Saving logistic regression model from training set 1
modelFileSave = open('models/model_m1', 'wb')
pickle.dump(model, modelFileSave)
modelFileSave.close()  
 