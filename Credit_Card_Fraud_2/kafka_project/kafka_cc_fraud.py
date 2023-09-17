import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk
from sklearn import preprocessing
import matplotlib.pyplot as plt
from kafka import KafkaConsumer
import json

st.header('Credit Card Fraud Detection ML Model side')

def std_df(dict1):
    total_fraud=0
    total_notfraud=0

    # Print fraud and not fraud till now
    #st.text(total_fraud)
    #st.text(total_notfraud)

    list1 = pd.DataFrame(dict1, index=[0])

    scaler = preprocessing.StandardScaler()

    X_std = scaler.fit_transform(list1)
#    return X_std


#def predict(X):

    X_df = pd.DataFrame(X_std, columns=list1.columns)

    model1 = open("CC_LR_under_model.pk", "rb")
    model_lr_under = pk.load(model1)

    #pred1 = model_lr_under.predict([[ X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7], X[8], X[9], X[10], X[11], X[12], X[13], X[14], X[15], X[16], X[17],X[18], X[19], X[20], X[21], X[22], X[23], X[24], X[25] ]])
    #pred1 = model_lr_under.predict([[ 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1 ]])
    X = X_df.values.reshape(1, -1)
    pred1 = model_lr_under.predict( X )


    
    print(pred1[0])
    #print(X.shape)
    if pred1[0] == 0:
        total_notfraud+=1
        print("******NOT FRAUD******")
        st.markdown(':green[      NOT FRAUD      ]')
    else:
        total_fraud+=1
        print("*******FRAUD*******")
        st.markdown(':red[      FRAUD      ]')


consumer = KafkaConsumer('cc', bootstrap_servers=['localhost:9092'], value_deserializer=lambda x: json.loads(x.decode('utf-8')))

for message in consumer:
    msg = message.value
    print('received: ', msg)
    st.text('Data recieved from bank: ')
    #st.text(msg)

    X = std_df(msg)
    #predict(X)


#X = std_df(record)

