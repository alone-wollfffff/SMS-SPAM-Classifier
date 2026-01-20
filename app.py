import streamlit as st
import pickle
import os
import requests
import io

# Load model from public URL
model_url = "https://drive.google.com/file/d/1SMHmW9z4YBAxic80_jokMqIatRGaQqMf/view?usp=sharing"
response = requests.get(model_url)
model = pickle.load(io.BytesIO(response.content))

vectorizer_url = "https://drive.google.com/file/d/1YLRGNiBU_jCI1xT10T6ie0BbBXGn9k6B/view?usp=sharing"
response = requests.get(vectorizer_url)
tfidf = pickle.load(io.BytesIO(response.content))

import string, nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()

#tfidf = pickle.load(open('vectorizer.pkl','rb'))
#model = pickle.load(open('model.pkl','rb'))

st.title("SMS Spam Classifier")
input_sms = st.text_area("Enter the Message...")

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation :
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
            
    return " ".join(y)


if st.button('Predict..'):
    # 1.PreProcess
    transformed_sms = transform_text(input_sms)
    # 2.Vectorize
    vector_input = tfidf.transform([transformed_sms]).toarray()
    # 3.Predict
    result = model.predict(vector_input)[0]
    # 4.Display
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')







