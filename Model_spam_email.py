import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import numpy as np

df = pd.read_csv("F:/University_HCMUTE/N3_HK2/AIOT/GiuaKy/emails.csv")
goal = df.Prediction
data = df.drop(['Prediction', 'Email No.'], axis='columns')
x_train, x_test, y_train, y_test = train_test_split(data ,goal, test_size = 0.2)
model = LogisticRegression()
model.fit(x_train, y_train)
#y_pred = model.predict(x_test)
#print(y_pred)

#accuracy = accuracy_score(y_test, y_pred)
# print("Độ chính xác:", accuracy)

def split_email(email_input):
    email_array = np.zeros(3000)
    words = email_input.split(' ')
    for word in words:
        word = word.lower()
        if word in data.columns:
            index = data.columns.get_loc(word)
            email_array[index] += 1
    email_array = email_array.reshape(1, -1)
    return email_array

def email_classification(email):
    email_pred = model.predict(split_email(email))
    return email_pred   
