import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
df = pd.read_csv('data.csv')

print(df)

df.info()

correlation = df['EngineSize'].corr(df['FuelConsumption'])

print(correlation)

plt.plot(df['EngineSize'],df['FuelConsumption'])
plt.xlabel('EngineSize')
plt.ylabel('FuelConsumption')
plt.title('EngineSize vs FuelConsumption')
plt.show()

x= df.drop(columns=['FuelConsumption'],axis=1)
y = df['FuelConsumption']

model = LinearRegression()

poly = PolynomialFeatures(degree=3)
x_square = poly.fit_transform(x)

model.fit(x_square,y)

result = model.predict(poly.fit_transform([[4.0]]))

print("Fuelconsumption for degree 3",result)