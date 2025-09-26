import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
import pickle
df = pd.read_csv('iris.csv')

encoder = LabelEncoder()

x = df.drop('species',axis =1)
y = df['species']

y = encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression()

model.fit(X_train,y_train)

with open('iris_model.bin', 'wb') as file:
    pickle.dump((model, encoder), file)

output = model.predict(pd.DataFrame([[5,3.6,1.4,0.2]],columns=['sepal_length','sepal_width','petal_length','petal_width']))

predicted_species = encoder.inverse_transform(output)

print(predicted_species)

y_pred = model.predict(X_test)

confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)

precision = precision_score(y_test, y_pred, average='weighted')
print("Precision:", precision)

f1score = f1_score(y_test, y_pred, average='weighted')
print("F1 Score:", f1score)