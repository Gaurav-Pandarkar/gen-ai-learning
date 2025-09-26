import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,f1_score


df = pd.read_csv('hearing_test.csv')

x = df.drop('test_result',axis=1)
y = df['test_result']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=23232323)

model = LogisticRegression()

model.fit(x_train,y_train)

output =model.predict(pd.DataFrame([[60,45]],columns=['age','physical_score']))

print("Predicted output for a 60 years old person with physical score of 45:",output)

y_pred = model.predict(x_test)

confusion = confusion_matrix(y_test,y_pred)
print("Confusion matrix:")
print(confusion)

accuracy = accuracy_score(y_test,y_pred)
print('Accuracy Score:')
print(accuracy)

precision = precision_score(y_test,y_pred,average='weighted')
print('######### Precision:')
print(accuracy)

f1score = f1_score(y_test,y_pred,average='weighted')
print('##### F1 Score')
print(f1score)

