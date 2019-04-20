# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 01:11:30 2018

@author: Jayant
"""
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import pickle
import re

stop_words = ('what', 'where','when','kab','\?','kaha','status','kya','kuthe','kute','hi','please','plz','status','update','why','ky','ahe ka','how')
questions = ('is','are','any','did')
center_stations = {'sion', 'kasara', 'diva', 'dolavli', 'kopar', 'kalyan', 'cst', 'karjat', 'airoli', 'chinchpokli', 'churchgate', 'asangaon', 'ulhas', 'dadar', 'tilak', 'thane', 'dombivali', 'neral', 'parel', 'andheri', 'mumbra', 'byculla', 'kalwa', 'masjid', 'ulhas nagar', 'bhivpuri road', 'palasdhari', 'khadavli', 'vikhroli', 'cstm', 'sandhurst', 'dombivli', 'curry road', 'lowjee', 'ambernath', 'kalva', 'shahad', 'vasind', 'bhandup', 'csmt', 'sandhurst road', 'atgaon', 'vithalwadi', 'badlapur', 'vidyavihar', 'thakurli', 'ghatkopar', 'diva jn', 'chunabhatti', 'kelavli', 'kanjur marg', 'khardi', 'titwala', 'bandra', 'ambivli', 'elphinstone', 'currey road', 'shelu', 'kanjur', 'matunga', 'kurla', 'vangani', 'khopoli', 'mulund', 'koparkhairne', 'nahur'}
train_type = ['slow', 'fast']

with open('tokenizer/tokenize_m1.pickle', 'rb') as handle:
    tokenize = pickle.load(handle)

def clean(text):
    #print("Removing stopwords and questions...")
    text = text.lower()
    splitted = word_tokenize(text)
    count = 0
    length = len(splitted)
    trainname = np.nan
    for idx, value in enumerate(splitted):
        if value in center_stations:
            count += 1
            if idx+1 < length:
                if splitted[idx+1] in train_type:
                    trainname = value + " " + splitted[idx+1] + " train"
       
    for word in text.split(' '):
        if count < 2:
            return get_sent_from(text), trainname, get_plt_no(text),  np.nan
        if value in stop_words:
            return get_sent_from(text), trainname, get_plt_no(text),  np.nan

    if text.startswith(questions):
        return get_sent_from(text), trainname, get_plt_no(text),  np.nan

    tex = pd.Series(text)
    numtext = tokenize.texts_to_matrix(tex)
    #print(tex.shape)
    return get_sent_from(text), trainname, get_plt_no(text),  numtext
    
def get_sent_from(text):
    match = re.search(r'sent from ([\w\.-]+)', text)
    if match:
        return match.group(1)
    else:
        return np.nan

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

def get_plt_no(text):
    match = re.search(r'([\d])', text)
    if match:
        return match.group(1) # The username (group 1) 
    else:
        return np.nan
    
    
    
#print("Final Text: ",clean("kalyan to dadar  (sent from dombivli)"))