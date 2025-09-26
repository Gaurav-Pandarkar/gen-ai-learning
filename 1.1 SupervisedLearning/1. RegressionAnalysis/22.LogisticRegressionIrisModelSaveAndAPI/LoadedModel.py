import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
from flask import Flask,request,jsonify

app = Flask(__name__)

with open('iris_model.bin','rb') as file:
    model, encoder = pickle.load(file)

output = model.predict(pd.DataFrame([[5,3.6,1.4,0.2]],columns=['sepal_length','sepal_width','petal_length','petal_width']))

predicted_species = encoder.inverse_transform(output)

print(predicted_species)

@app.route("/predict",methods=['POST'])
def predict_iris():
    data = request.get_json()
    sepal_length = data["sepal_length"]
    sepal_width = data["sepal_width"]
    petal_length = data["petal_length"]
    petal_width = data["petal_width"]

       
    input_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])

    output = model.predict(input_df)
    predicted_species = encoder.inverse_transform(output)[0]

    return jsonify({"prediction": predicted_species})


if __name__=='__main__':
    app.run(debug=True)
