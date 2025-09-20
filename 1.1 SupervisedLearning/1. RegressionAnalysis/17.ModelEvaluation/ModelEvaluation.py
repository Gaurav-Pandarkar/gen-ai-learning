import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import numpy as np

data = pd.read_csv('SpendingData.csv')
x = data.drop("Spendings",axis =1)
y = data["Spendings"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=10000)

model = LinearRegression()
model.fit(x_train,y_train)

train_score = model.score(x_train,y_train)
test_score = model.score(x_test,y_test)

print(f"Train r2 score:{train_score}")
print(f"Test r2 score:{test_score}")

y_pred = model.predict(x_test)

mae  = mean_absolute_error(y_test,y_pred)

mse = mean_squared_error(y_test,y_pred)

rmse = np.sqrt(mse)   ## root mean square error

mape = np.mean(np.abs((y_test-y_pred)/y_test))*100  ##mean absolute percentage error

print(f"MAE:{mae}")
print(f"MSE :{mse}")
print(f"RMSE:{rmse}")
print(f"MAPE :{mape}")