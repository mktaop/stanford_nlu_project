#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 14:11:02 2023

@author: avi_patel
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import re
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import tqdm
from tqdm import tqdm

data=pd.read_csv('/Users/avi_patel//Downloads/complaints-2023-02-26_13_52.csv') #from cfpb.com
data.info()

ndata=pd.DataFrame(data['Consumer complaint narrative'])
ndata=ndata.rename(columns={ndata.columns[0]:'comment'})
ndata.comment=ndata.comment.astype(str)
ndata.comment=ndata.comment.str.lower()
stop_words=stopwords.words('english')
new_stopwords=["xxxx","xxxxxxxx"]
stop_words.extend(new_stopwords)
#ndata.comment=ndata['comment'].apply(lambda x:''.join(
    #[word for word in x.split() if word not in (stop_words)]))
ndata2=ndata.sample(frac=.01, replace=True, random_state=1) # 1% sample to play with
play=ndata.tail(300)

for i in range(len(play)):
    text=' '.join(play.comment)
    sent_list=nltk.sent_tokenize(text)
    sent_list = [''.join([char for char in line if char.isalnum() or char == ' ']) for line in sent_list]

vect = CountVectorizer(stop_words=None, token_pattern=r"(?u)\b\w+\b")
X = vect.fit_transform(sent_list)
uniq_wrds = vect.get_feature_names()
n = len(uniq_wrds)
co_mat = np.zeros((n,n))

window_len = 10
def update_co_mat(x):   
    # Get all the words in the sentence and store it in an array wrd_lst
    wrd_list = x.split(' ')
    wrd_list = [ele for ele in wrd_list if ele.strip()]
    #print(wrd_list)
    
    # Consider each word as a focus word
    for focus_wrd_indx, focus_wrd in enumerate(wrd_list):
        focus_wrd = focus_wrd.lower()
        # Get the indices of all the context words for the given focus word
        for contxt_wrd_indx in range((max(0,focus_wrd_indx - window_len)),(min(len(wrd_list),focus_wrd_indx + window_len +1))):                        
            # If context words are in the unique words list
            if wrd_list[contxt_wrd_indx] in uniq_wrds:
                
                # To identify the row number, get the index of the focus_wrd in the uniq_wrds list
                co_mat_row_indx = uniq_wrds.index(focus_wrd)
                
                # To identify the column number, get the index of the context words in the uniq_wrds list
                co_mat_col_indx = uniq_wrds.index(wrd_list[contxt_wrd_indx])
                                
                # Update the respective columns of the corresponding focus word row
                co_mat[co_mat_row_indx][co_mat_col_indx] += 1

for sentence in tqdm(sent_list):
    update_co_mat(sentence)
df=pd.DataFrame(co_mat, columns=uniq_wrds, index=uniq_wrds)