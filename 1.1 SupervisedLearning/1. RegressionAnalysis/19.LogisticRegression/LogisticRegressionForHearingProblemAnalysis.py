import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

df  = pd.read_csv('hearing_test.csv')

print(df.corr())

hearing_problem = df[df['test_result']==1]

no_hearing_problem = df[df['test_result']==0]

plt.scatter(hearing_problem['age'],hearing_problem['physical_score'],color='red',marker ='.')
plt.scatter(no_hearing_problem['age'],no_hearing_problem['physical_score'],color='green',marker ='.')
plt.show()

x= df.drop('test_result',axis=1)
y=df['test_result']

model = LogisticRegression()
model.fit(x,y)

output = model.predict(pd.DataFrame([[20,50]],columns=['age','physical_score']))

print(output)