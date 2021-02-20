# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import spacy
import json
import re
import scipy
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from nltk.stem import PorterStemmer
from string import punctuation
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from flask import Flask,render_template,url_for,request
import pickle
import joblib
import gensim
import smart_open
from gensim.models import Word2Vec

app = Flask(__name__)

model = gensim.models.KeyedVectors.load_word2vec_format('C:\\Users\\Suraj\\GoogleNews-vectors-negative300.bin.gz', binary=True)

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
    
    return xheaddf

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods=['GET','POST'])
def predict():
    print('inside predict')
    if(request.method == 'POST'):
        headline = request.form['rawtext']
        predictdf = feature_preparation(headline)
        modelchoice = request.form['modelchoice']
        if(modelchoice == 'nb'):
            news_clf=pickle.load(open('static/models/GNB.pkl','rb'))
            
        elif(modelchoice == 'rf'):
            news_clf=pickle.load(open('static/models/RFC.pkl','rb'))
            
        prediction_lables = {'LIFESTYLE AND WELLNESS':0,'POLITICS':1,'SPORTS AND ENTERTAINMENT':2,'TRAVEL-TOURISM AND ART-CULTURE':3}
        prediction = news_clf.predict(predictdf)
        final_result = get_keys(prediction,prediction_lables)
        
    return render_template('index.html',headline = headline.upper(),final_result = final_result)


if __name__ == '__main__':
	app.run(debug=True)