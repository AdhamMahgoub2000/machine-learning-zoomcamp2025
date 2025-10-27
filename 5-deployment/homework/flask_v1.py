import pickle
from flask import Flask, request, jsonify


app = Flask("lead_scoring_model")

with open('pipeline_v1.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

@app.route('/', methods=['GET'])
def hello_world():
    return "Hello, Lead Scoring Model!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    X_vectorized = dv.transform([data])
    y_pred = model.predict_proba(X_vectorized)[:, 1]
    print(f"Predicted probability: {y_pred}")
    return jsonify({'lead_scoring_probability': float(y_pred),
                    'lead_scoring_prediction': int(y_pred >= 0.5)
                   })
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9000)