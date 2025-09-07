import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pickle


df = pd.read_csv('data.csv')

print(df)

df.info()

correlation = df['Speed'].corr(df['BrakingDistance'])

print(correlation)

plt.plot(df['Speed'], df['BrakingDistance'])
plt.xlabel('Speed')
plt.ylabel('BrakingDistance')
plt.title('Speed vs BrakingDistance')
plt.show()


x = df.drop(columns=['BrakingDistance'],axis=1)
y = df['BrakingDistance']

poly = PolynomialFeatures(degree=5)  
x_square = poly.fit_transform(x)

print(x_square)

model = LinearRegression()
model.fit(x_square,y)


with open ('car_prediction.bin','wb') as file:  #save model in car_prediction.bin  this file
    pickle.dump(model,file)

output = model.predict(poly.fit_transform([[120]]))

print("Degree 5 for 120",output)