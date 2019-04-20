# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 22:01:05 2018

@author: Jayant
"""

import glob
from nltk.tokenize import word_tokenize
import pandas as pd
from collections import Counter

# =============================================================================
# train = pd.read_json('Hackathon ML data/20180701_CENTRAL_LINE.txt');
# train = train.T;
# print(train.shape)
# =============================================================================

path =r'Hackathon ML data'
allFiles = glob.glob(path + "/*.txt")

list_files = []
count = 0

for file_ in allFiles:
    if count == 4:
        break
    df = pd.read_json(file_)
    list_files.append(df)
    count += 1

train = pd.concat(list_files, axis = 1, ignore_index = True)
train = train.T;
print(train.shape)


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

final_data = pd.DataFrame();

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
questions = ('is','are','any','did')

# =============================================================================
# def torf(text):
#     if any(ext in text for ext in whlist):
#     #if not text.str.contains('|'.join(whlist)):
#         return False
#     elif text.startswith(questions):
#         return False
#     return True
# 
# train['bool'] = train.m.apply(torf)
# print(train['bool'].value_counts())
# =============================================================================
station_list = set(station_list)
print(~cleaned.m.str.contains('|'.join(whlist)))
cleaned['category'] = ~cleaned.m.str.contains('|'.join(whlist))
cleaned['category'] = cleaned['category'].map({True:'',False:'nan'})
print(cleaned.shape)

print("Removing stating with...")
questions = ('is','are','any','did')
cleaned = cleaned[~cleaned.m.str.startswith(questions)]
print(cleaned.shape)
#print(len(station_list))

print("Removing stations...")
for index, row in cleaned.iterrows():
    splitted = word_tokenize(row.m)
# =============================================================================
#     count = 0
#     for element in splitted:
#         if element in station_list:
#             count += 1
# =============================================================================
    count = len([i for i in splitted if i in station_list])
    if count < 2:
        cleaned.drop(index, inplace=True)

print(cleaned.shape)   
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
        return 'nan'
    
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
    
    return 'nan'

cleaned['category']=cleaned['m'].apply(make_categories)
print(cleaned['category'].value_counts())

cleaned['delay_time']='nan'
#dig=cleaned.iloc[0,0].extract('(\d+)')
#print(dig)
def get_delay_time(cat, m):
    print(cat)
    if cat == "delay":
        text=m
        match = re.search(r'([\d]+) min', text)
        if(match):
            print('hi')
            return match.group(1)
        match = re.search(r'([\d]+)min', text)
        if(match):
            print('hi')
            return match.group(1)
        return 'nan'
    return 'nan'
cleaned['delay_time'] = cleaned.apply(lambda row: get_delay_time(row['category'],row['m']), axis=1)

final_train = train.copy()
final_train['category'] = 'nan'
for index, row in train.iterrows():
    if index in cleaned.index:
        final_train.iloc[index]['category'] = cleaned.iloc[index].category
    else:
        final_train.iloc[index]['category'] = 'nan'
        
final_train['category'] = cleaned['category']
new_df = cleaned[['time','category']].copy()


train.describe()
final_train = pd.merge(train, cleaned])

# =============================================================================
# match = re.search(r'left', "5:43 titwala fast train lefted from platform no 4 (sent from dadar stn.)")
# if match:
#     print('hi') # The username (group 1) 
# else:
#     print('bye')
# 
# =============================================================================


























    
 