import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('salary_data (1).csv')

x = df.drop('Salary',axis =1)
y = df['Salary']

column_transformer = ColumnTransformer(
    transformers=[
        ('onehot',OneHotEncoder(sparse_output=True,drop='first'),['Title'])
    ],
    remainder ='passthrough'
)

#print(column_transformer)

transformed_values = column_transformer.fit_transform(x)

print(transformed_values)
# output
# [[ 0.  0.  1.]
#  [ 0.  0.  2.]
#  [ 0.  0.  3.]
#  [ 0.  0.  4.]
#  [ 0.  0.  5.]
#  [ 0.  1.  1.]
#  [ 0.  1.  2.]
#  [ 0.  1.  3.]
#  [ 0.  1.  4.]
#  [ 0.  1.  5.]
#  [ 1.  0.  2.]
#  [ 1.  0.  3.]
#  [ 1.  0.  5.]
#  [ 1.  0.  7.]
#  [ 1.  0. 10.]]

transformed_features = pd.DataFrame(transformed_values,columns=column_transformer.get_feature_names_out())

print(transformed_features)

#     onehot__Title_Project Manager  onehot__Title_Software Engineer  remainder__Experience
# 0                             0.0                              0.0                    1.0
# 1                             0.0                              0.0                    2.0
# 2                             0.0                              0.0                    3.0
# 3                             0.0                              0.0                    4.0
# 4                             0.0                              0.0                    5.0
# 5                             0.0                              1.0                    1.0
# 6                             0.0                              1.0                    2.0
# 7                             0.0                              1.0                    3.0
# 8                             0.0                              1.0                    4.0
# 9                             0.0                              1.0                    5.0
# 10                            1.0                              0.0                    2.0
# 11                            1.0                              0.0                    3.0
# 12                            1.0                              0.0                    5.0
# 13                            1.0                              0.0                    7.0
# 14                            1.0                              0.0                   10.0

model = LinearRegression()
model.fit(transformed_features,y)

data_to_predict = pd.DataFrame([['Project Manager',2]],columns= ['Title','Experience'])
print(data_to_predict)

# [15 rows x 3 columns]
#              Title  Experience
# 0  Project Manager           2

new_data_transformed = column_transformer.transform(data_to_predict)
print(new_data_transformed)

# [[1. 0. 2.]]

data_frame_to_predict = pd.DataFrame(new_data_transformed, columns=column_transformer.get_feature_names_out())
y_pred = model.predict(data_frame_to_predict)

print(y_pred)

# [63833.33333333]