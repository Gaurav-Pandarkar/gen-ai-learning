import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df = pd.read_csv('Mall_Customers.csv')
df = df.drop(['CustomerID','Gender','Age'],axis=1)
df.rename(columns={"Annual Income (k$)":"Income","Spending Score (1-100)":"Spending"},inplace=True)

wss = []
clusters_range = range(1,11)

for k in clusters_range:
    model = KMeans(n_clusters=k,random_state=1234)
    model.fit(df)
    wss.append(model.inertia_)

plt.figure(figsize=(8,5))
plt.plot(clusters_range,wss,marker='o')
plt.title("Elbow method for optimal k")
plt.xlabel("Number of clusters {K}")
plt.ylabel("WSS (Inertia)")
plt.grid(True)
plt.show()
