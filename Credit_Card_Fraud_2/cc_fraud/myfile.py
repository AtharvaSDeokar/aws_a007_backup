import statistics

import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk
from sklearn import preprocessing
import matplotlib.pyplot as plt


def std_df(df):
	from sklearn.preprocessing import StandardScaler
	stdr = preprocessing.StandardScaler()
	X_std = stdr.fit_transform(df)
	
	return X_std

list_pred = []
def predict(X):

    model1 = open("CC_LR_under_model.pk", "rb")
    model3 = open("CC_LR_SMOTE.pk", "rb")
    model2 = open("CC_LR_over_model.pk", "rb")

    model_smote = pk.load(model3)
    model_lr_under = pk.load(model1)
    model_lr_over = pk.load(model2)



    for x in X:

        v1=x[1]
        v2=x[2]
        v3=x[3]
        v4=x[4]
        v5=x[5]
        v6=x[6]
        v7=x[7]
        v8=x[8]
        v9=x[9]
        v10=x[10]
        v11=x[11]
        v12=x[12]
        v13=x[13]
        v14=x[14]
        v15=x[15]
        v16=x[16]
        v17=x[17]
        v18=x[18]
        v19=x[19]
        v20=x[20]
        v21=x[21]
        v22=x[22]
        v23=x[23]
        v24=x[24]
        v25=x[25]
        v26=x[26]
        
        pred1 = model_lr_under.predict([[ x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17], x[18], x[19], x[20], x[21], x[22], x[23], x[24], x[25], x[26] ]])
        pred2 = model_lr_over.predict([[ x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17], x[18], x[19], x[20], x[21], x[22], x[23], x[24], x[25], x[26] ]])
        pred3 = model_smote.predict([[ x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17], x[18], x[19], x[20], x[21], x[22], x[23], x[24], x[25], x[26] ]])


        ans = ( (pred1 + pred2 + pred3) / 3)
        
        if ans > 0.5:
            list_pred.append(1)
        
        else:
            list_pred.append(0)




st.header("Credit Card fraud Detection with Machine Learning")

st.subheader("Upload Dataset")
data_file = st.file_uploader("Upload CSV", type=["csv"])
if data_file is not None:
    file_details = {"filename": data_file.name, "filetype": data_file.type, "filesize": data_file.size}

    st.write(file_details)
    df = pd.read_csv(data_file)
    st.dataframe(df)
    X = std_df(df)
    print(predict(X))

    st.subheader("Predicted output is: ")

    ser_list = pd.Series( list_pred )
    st.text(ser_list.value_counts())
    not_fraud, fraud = ser_list.value_counts()
    st.text(f"not fraud : {not_fraud}\t fraud : {fraud}")
    st.text(f"Actual fraud cases are 20")
    
    # Pie chart
   
    labels = "Not Fraud", "FRAUD!!"
    sizes = [not_fraud, fraud]
    explode = (0, 0.1)   # Explode only fraud cases
    
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%.2f', shadow=True, startangle=90)
    ax1.axis('equal')  # equal aspect ratio ensures that pie is drawn as a circle
    
    st.pyplot(fig1)
    
    
    
