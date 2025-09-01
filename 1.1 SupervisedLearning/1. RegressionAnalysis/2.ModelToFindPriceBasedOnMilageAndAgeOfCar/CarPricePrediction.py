import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

df = pd.read_csv('car_price_data.csv')

print(df.columns)

df.info()

crr_age_price = df["Age"].corr(df['Price'])
crr_mileage_price = df['Mileage'].corr(df['Price'])

print("Correlation Between Age and Price",crr_age_price)
print("Correlation Between Mileage and Price",crr_mileage_price)

plt.scatter(df['Mileage'], df['Price'])
plt.xlabel('Mileage')
plt.ylabel('Price')
plt.title('Mileage vs Price')
plt.show()

plt.scatter(df['Age'],df['Price'])
plt.xlabel('Age')
plt.ylabel('Price')
plt.title('Age vs Price')
plt.show()

model = LinearRegression()

x= df.drop('Price',axis=1)
y=df['Price']

print(x)
print(y)

model.fit(x,y)

input_data = pd.DataFrame([[5,50000]],columns=['Age','Mileage'])

print(input_data)

predicted_price = model.predict(input_data)

print(f"Predicted Price for a car with 5 years of age and 50000 mileage :{predicted_price[0]}")