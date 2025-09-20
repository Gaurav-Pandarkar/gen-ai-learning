import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("SpendingData.csv")

x = data.drop("Spendings",axis=1)
y = data["Spendings"]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1000)

model = LinearRegression()

model.fit(x_train,y_train)

train_score = model.score(x_train,y_train)
test_score = model.score(x_test,y_test)

print(f"Train Score:{train_score}")
print(f"Test Score:{test_score}")
