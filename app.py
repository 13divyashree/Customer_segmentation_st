import pandas as pd
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from kneed import kneeLocator
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score


st.set_page_config(page_icon="👥",page_title="Customer Segmentation",layout="wide")
 

file=st.file_uploader(" ",type='csv')
df=None
if file:
    df=pd.read_csv(file)


with st.sidebar:
    st.title("Customer Segmentation")
    if df is not None:
        features=st.multiselect("SELECT FEATURE",options=df.columns,default=["Annual Income (k$)","Spending Score (1-100)"])
        df=df.loc[:features]

def preprocessing(df):
    encoder=LabelEncoder()
    for col in df.columns:
        if df[col].dtype==object:
            df[col]=encoder.fit_transform(df[col])

def elbow():
    out=[]
    k_values=range(1,11)
    for i in k_values:
        model=KMeans(n_clustres=i)
        model.fit(df)
        out.append(model.inertia_)
    KL=KneeLocator(k_values,out,curve="convex",direction="decreasing")
    df1=pd.DataFrame({"k_values":k_values,"inertia":out})
    st.header("ELBOW CURVE")
    fig=st.line_chart(data=df1,x="k_values",y="inrtia")
    return KL.elbow



if df is not None:
    st.subheader("Sample data")
    st.write(df.sample(10))
    preprocessing(df)



