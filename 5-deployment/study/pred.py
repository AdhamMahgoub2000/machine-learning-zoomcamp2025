import pickle
from flask import Flask , request , jsonify


with open('model.bin','rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('conversion_predictor')

@app.route('/', methods=['GET'])
def hello_world():
    return "Hello_World!"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        customer = request.get_json()

        X_vectorized = dv.transform([customer])
        y_pred = model.predict_proba(X_vectorized)[:, 1]

        result = {'conversion_probability': float(y_pred),
                'conversion_prediction': int(y_pred >= 0.5)
                }
        print(result)
        return jsonify(result)
    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)