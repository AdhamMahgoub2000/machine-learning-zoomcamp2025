import requests

url = 'http://localhost:9000/predict'

customer = {
    "lead_source":"paid_ads",
    "industry": "retail",
    "number_of_courses_viewed": 5,
    "annual_income": 83843.0,
    "employment_status": "NaN",
    "location": "australia",
    "interaction_count": 1,
    "lead_score": 0.87
}

response = requests.post(url, json=customer)
print("Status:", response.status_code)
print("Response text:", response.text)