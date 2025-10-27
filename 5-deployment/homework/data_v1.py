import requests


x = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}
url = 'http://0.0.0.0:9000/predict'
response = requests.post(url, json=x)
print(response.status_code)
print(response.json())