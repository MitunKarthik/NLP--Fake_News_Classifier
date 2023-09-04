#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


# In[2]:


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
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    res = " ".join(x for x in text if x.isalpha())
    y.append(res)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# In[3]:


cv = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


# In[4]:


st.title("Fake News Classifier")


# In[5]:


input = st.text_input("Enter the text")


# In[6]:


if st.button('Predict'):
    #1. Preprocess
    transform_news = transform_text(input)
    #2. Vectorize
    news_vector = cv.transform([transform_news])
    #3. Predict
    result = model.predict(news_vector)
    #4. Display
    if result == 1:
        st.header("Fake")
    else:
        st.header("True")


# In[ ]:




