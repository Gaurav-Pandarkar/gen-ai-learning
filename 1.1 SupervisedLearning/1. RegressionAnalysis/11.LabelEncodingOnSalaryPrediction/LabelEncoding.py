import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression


df = pd.read_csv('SalaryData.csv')

x= df.drop('Salary',axis=1)
y= df['Salary']

encoder = LabelEncoder()

x['Title'] = encoder.fit_transform(x['Title'])

model = LinearRegression()

model.fit(x,y)

output = model.predict(pd.DataFrame([[8,1]],columns = ['Experience','Title']))

print(output)