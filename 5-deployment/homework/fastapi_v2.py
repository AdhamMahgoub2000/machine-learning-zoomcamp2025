from fastapi import FastAPI
import pickle

app = FastAPI()
with open('pipeline_v1.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

@app.get("/")
def read_root():
    return "Hello, Lead Scoring Model!"

@app.post("/predict")
def predict(data: dict):
        x_vectorized = dv.transform([data])
        y_pred = model.predict_proba(x_vectorized)[:, 1]
        return {
            'lead_scoring_probability': float(y_pred),
            'lead_scoring_prediction': int(y_pred >= 0.5)
        }