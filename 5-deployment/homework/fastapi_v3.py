from fastapi import FastAPI
import pickle
import uvicorn


with open('/code/pipeline_v2.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

app = FastAPI()

@app.get("/")
def read_root():
    return "Hello, Lead Scoring Model!"


@app.post("/predict")
def predict(data: dict):
        y_pred = pipeline.predict_proba([data])[:, 1]
        return {
            'lead_scoring_probability': float(y_pred),
            'lead_scoring_prediction': int(y_pred >= 0.5)
        }

if __name__ == '__main__':

    uvicorn.run(app, host='0.0.0.0', port=9000)