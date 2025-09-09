import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer

df = pd.read_csv('salary_data.csv')


imputer = KNNImputer(n_neighbors=3)

df['Salary'] = imputer.fit_transform(df[['Salary']])

print(df)

x = df.drop(['Salary'],axis=1)
y = df['Salary']

model = LinearRegression()

model.fit(x,y)

result = model.predict(pd.DataFrame([3],columns =['Experience']))

print(result)