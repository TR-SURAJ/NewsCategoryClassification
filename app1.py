# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 20:17:02 2021

@author: Suraj
"""
import numpy as np
import pandas as pd
import spacy
import streamlit as st 
import json
import re
import scipy
import os
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from nltk.stem import PorterStemmer
from string import punctuation
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import pickle
import joblib
import gensim
import smart_open
from gensim.models import Word2Vec


model = gensim.models.KeyedVectors.load_word2vec_format('C:\\Users\\Suraj\\GoogleNews-vectors-negative300.bin.gz', binary=True)


def load_prediction_models(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model

def get_keys(val,my_dict):
    for key,value in my_dict.items():
        if val == value:
            return key
        
SPECIAL_TOKENS = {
    'quoted': 'quoted_item',
    'non-ascii': 'non_ascii_word',
    'undefined': 'something'
}


def clean(text, stem_words=True):

    
    if pd.isnull(text):
        return ''

    if type(text) != str or text=='':
        return ''
    ps = PorterStemmer()
    
    text = re.sub("\'s", " ", text) 
    text = re.sub(" whats ", " what is ", text, flags=re.IGNORECASE)
    text = re.sub("\'ve", " have ", text)
    text = re.sub("can't", "can not", text)
    text = re.sub("n't", " not ", text)
    text = re.sub("i'm", "i am", text, flags=re.IGNORECASE)
    text = re.sub("\'re", " are ", text)
    text = re.sub("\'d", " would ", text)
    text = re.sub("\'ll", " will ", text)
    text = re.sub("e\.g\.", " eg ", text, flags=re.IGNORECASE)
    text = re.sub("b\.g\.", " bg ", text, flags=re.IGNORECASE)
    text = re.sub("(\d+)(kK)", " \g<1>000 ", text)
    text = re.sub("e-mail", " email ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?U\.S\.A\.", " America ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?United State(s)?", " America ", text, flags=re.IGNORECASE)
    text = re.sub("\(s\)", " ", text, flags=re.IGNORECASE)
    text = re.sub("[c-fC-F]\:\/", " disk ", text)
    
    # remove comma between numbers, i.e. 15,000 -> 15000
    
    text = re.sub('(?<=[0-9])\,(?=[0-9])', "", text)
    text = re.sub('\$', " dollar ", text)
    text = re.sub('\%', " percent ", text)
    text = re.sub('\&', " and ", text)
    
       # Return a list of words
    text = ' '.join([word for word in text.split(" ") if word not in stop_words]).lower()
    #text = ' '.join([ps.stem(word) for word in text])
    return text

def sent2vec(s):
    print('inside the sent2vec')
    words = str(s).lower() 
    words = word_tokenize(words) 
    words = [w for w in words if not w in stop_words] 
    words = [w for w in words if w.isalpha()] 
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M) 
    v = M.sum(axis=0) 
    return v / np.sqrt((v ** 2).sum())


def feature_preparation(headline):
    print('inside feature preparation')
    
    a = {'headline': headline}
    testdf = pd.DataFrame(a,columns = ['headline'],index=[0])
    testdf['headline'] = testdf['headline'].apply(clean)
    
    headline_vector = np.zeros((testdf.shape[0], 300))
    for i, q in enumerate(testdf.headline.values):
        headline_vector[i, :] = sent2vec(q)

    xheaddf = pd.DataFrame(headline_vector)
    print(xheaddf)
    
    return xheaddf



        
def main():
	
    st.title("News Category Classifier")
    html_temp = """<div style="background-color:blue;padding:10px"><h1 style="color:white;text-align:center;">Streamlit ML App</h1></div>"""
    st.markdown(html_temp,unsafe_allow_html=True)
    headline = st.text_area("Enter News Here")
    print(headline)
    
    all_ml_models = ["Gaussian Naive Bayes","Random Forest"]
    model_choice = st.selectbox("Select Model",all_ml_models)
    
    prediction_lables = {'LIFESTYLE AND WELLNESS':0,'POLITICS':1,'SPORTS AND ENTERTAINMENT':2,'TRAVEL-TOURISM AND ART-CULTURE':3}
    
    if(st.button("Classify")):
        print('inside classify')
        st.text("Original Text:\n{}".format(headline))
        predictdf = feature_preparation(headline)
        
        if(model_choice == "Gaussian Naive Bayes"):
            print('inside GNB')
            predictor = load_prediction_models("static/models/GNB.pkl")
            prediction = predictor.predict(predictdf)
            print(prediction)
            
        
        elif(model_choice == "Random Forest"):
            predictor = load_prediction_models("static/models/RFC.pkl")
            prediction = predictor.predict(predictdf)
            print(prediction)
        
        final_result = get_keys(prediction,prediction_lables)
        print(final_result)
        st.success("News Categorized as:: {}".format(final_result))
    
if __name__ == '__main__':
    main()