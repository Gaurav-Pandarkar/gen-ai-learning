import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("student_data.csv")

X = df.drop("pass_fail", axis=1)
y = df["pass_fail"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression()
model.fit(X_train, y_train)

output = model.predict(pd.DataFrame([[3,7]],columns=['study_hours','sleep_hours']))

predicted_output = encoder.inverse_transform(output)

print(predicted_output)