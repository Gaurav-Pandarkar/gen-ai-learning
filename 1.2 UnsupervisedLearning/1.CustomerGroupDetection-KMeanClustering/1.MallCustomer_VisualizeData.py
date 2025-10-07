import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Mall_Customers.csv")
df = df.drop(['CustomerID','Gender','Age'],axis=1)
df.rename(columns={'Annual Income (k$)':'Income','Spending Score (1-100)':'Spending'},inplace=True)

plt.scatter(df['Income'],df['Spending'],color='blue')
plt.title('Customer Income vs Spending score')
plt.xlabel('Annual income (k$)')
plt.ylabel('Spending score (1-100)')
plt.grid(True)
plt.show()