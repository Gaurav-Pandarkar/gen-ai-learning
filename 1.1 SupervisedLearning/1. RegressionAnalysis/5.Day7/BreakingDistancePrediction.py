import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


df = pd.read_csv('data.csv')

# print(df)

# df.info()

correlation = df['Speed'].corr(df['BrakingDistance'])

# print(correlation)

# plt.plot(df['Speed'], df['BrakingDistance'])
# plt.xlabel('Speed')
# plt.ylabel('BrakingDistance')
# plt.title('Speed vs BrakingDistance')
# plt.show()


x = df.drop(columns=['BrakingDistance'],axis=1)
y = df['BrakingDistance']

poly = PolynomialFeatures(degree=5)  # degree 5 is more closely to correct prediction
x_square = poly.fit_transform(x)

# print(x_square)

model = LinearRegression()
model.fit(x_square,y)

output = model.predict(poly.fit_transform([[115]]))

print("Degree 5 for 115",output)