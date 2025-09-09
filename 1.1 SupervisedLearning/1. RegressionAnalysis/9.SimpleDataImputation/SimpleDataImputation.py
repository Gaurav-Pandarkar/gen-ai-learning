import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

df = pd.read_csv('salary_data.csv')
# 4 Stratergies of imputation using simple imputer
#imputer = SimpleImputer(strategy='mean')
#imputer = SimpleImputer(strategy='median')
#imputer = SimpleImputer(strategy='most_frequent')
imputer = SimpleImputer(strategy = 'constant',fill_value=12000)

df['Salary'] = imputer.fit_transform(df[['Salary']])

print(df)

x = df.drop('Salary',axis=1)
y = df['Salary']

model = LinearRegression()

model.fit(x,y)

result = model.predict(pd.DataFrame([10.5],columns=['Experience']))

print(result)