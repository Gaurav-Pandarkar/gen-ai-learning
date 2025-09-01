import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

df = pd.read_csv('Advertising.csv')

print(df.columns)

df.info()

crr_TV_sales = df["TV"].corr(df['sales'])
crr_radio_sales = df['radio'].corr(df['sales'])
crr_newspaper_sales = df['newspaper'].corr(df['sales'])

print("Correlation between TV and sales is :",crr_TV_sales)
print("Correlation between radio and sales is :",crr_radio_sales)
print("Correlation between newspaper and sales is :",crr_newspaper_sales)

plt.scatter(df['TV'], df['sales'])
plt.xlabel('TV')
plt.ylabel('sales')
plt.title('TV vs sales')
plt.show()

plt.scatter(df['radio'], df['sales'])
plt.xlabel('radio')
plt.ylabel('sales')
plt.title('radio vs sales')
plt.show()

plt.scatter(df['newspaper'], df['sales'])
plt.xlabel('newspaper')
plt.ylabel('sales')
plt.title('newspaper vs sales')
plt.show()

model = LinearRegression()

x = df.drop(['sales','newspaper'],axis=1)
y = df['sales']

print(x)
print(y)

model.fit(x,y)

input_data = pd.DataFrame([[50.23,30.45]],columns=['TV','radio'])

print(input_data)

predicted_sale = model.predict(input_data)

print(f"Predicted Sales using sales of TV=50.23 and radio=30.45 :{predicted_sale[0]}")