import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('salary_data.csv')

# x = df.drop('Salary', axis=1)
x = df['Experience']
x=x.values.reshape(-1,1)
y = df['Salary']

model = LinearRegression()

model.fit(x, y)

salaries = model.predict(pd.DataFrame([[15],[16]],columns=['Experience']))
print("Salary of 15 years of experince is : ",salaries[0])
print("Salary of 15 years of experince is : ",salaries[1])


print("Coefficient is :", model.coef_[0])
print("Intercept is :", model.intercept_)


m = model.coef_[0]
c = model.intercept_

salary = m * 15 + c

print("Salary of 15 years of experience using formula is :",salary)