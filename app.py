import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import numpy as np

import string
def process(obj):
    obj = obj.lower()
    ans = []
    obj = word_tokenize(obj)
    for i in obj:
        if i.isalnum():
            ans.append(i)
    obj = ans[:]
    ans.clear()
    stop_words = set(stopwords.words('english'))
    for i in obj:
        if i not in stop_words and i not in string.punctuation:
            ans.append(i)
    ps = PorterStemmer()
    obj = ans[:]
    ans.clear()
    for i in obj:
        k = ps.stem(i)
        ans.append(k)
    return " ".join(ans)
vecterizer = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('spam message classifier')
sms = st.text_area("enter the message")
if st.button('pridict'):
   smsTranformed = process(sms)

   smsVector = vecterizer.transform([smsTranformed])
   predect = model.predict(smsVector)[0]
   if(predect == 1):
     st.header('spam')
   if(predect == 0):
     st.header('not spam')
