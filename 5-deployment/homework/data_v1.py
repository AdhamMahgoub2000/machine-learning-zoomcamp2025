import requests

url = 'http://0.0.0.0:9000/predict'

x = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

response = requests.post(url, json=x)

print(response.json())