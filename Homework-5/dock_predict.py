import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from flask import Flask
from flask import request
from flask import jsonify

#Loading the model 
file_input_model = 'model2.bin'
file_input_dv = 'dv.bin'

with open(file_input_model, 'rb') as file_inp_md:
    model = pickle.load(file_inp_md)
    
with open(file_input_dv, 'rb') as file_inp_dv:
    dv = pickle.load(file_inp_dv)

print('Dict and model files are loaded. ')

#Initializing flask app
app = Flask('ccdata')

@app.route('/predict', methods=['POST'])
def predict():
    record = request.get_json()

    X = dv.transform(record)
    y_pred = model.predict_proba(X)[0, 1]
    pred = y_pred >= 0.5

    result = {
        'predicted probability': float(y_pred),
        'decision': bool(pred)
    }
    
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug = True, host='0.0.0.0', port=9696)