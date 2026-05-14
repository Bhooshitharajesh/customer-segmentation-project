import numpy as np
import pandas as pd
import datetime as dt
import joblib

df = pd.read_csv("data/online_retail_II.csv")

df=df.dropna(subset=["Customer ID"])

df=df[df["Quantity"]>0]

df=df[df["Price"]>0]

df['Total price']=df["Quantity"]*df["Price"]

df["InvoiceDate"]=pd.to_datetime(df["InvoiceDate"])

today=dt.datetime(2011,12,10)
df.groupby("Customer ID")
rfm=df.groupby("Customer ID").agg({
    "InvoiceDate":lambda x:(today-x.max()).days,
    "Invoice":"count",
    "Total price":"sum"})
rfm.columns=['Recency','Frequency','Monetary']

rfm["segment"]=rfm["Monetary"].apply(lambda x:'HIGH' if x>1000 else 'low')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

rfm_numeric = rfm[['Recency', 'Frequency', 'Monetary']]

scaler=StandardScaler()
rfm_scaled=scaler.fit_transform(rfm_numeric)

kmeans=KMeans(n_clusters=3)
rfm['Cluster']=kmeans.fit_predict(rfm_scaled)

rfm["customer_type"]=rfm["Cluster"].map({
    1:"high value",
    0:"regular",
    2:"low value"})

joblib.dump(kmeans,"kmeans.pkl")
joblib.dump(scaler,"scaler.pkl")