import requests


url = 'http://0.0.0.0:8080/predict'

x = {"lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0}

response = requests.post(url, json=x)
print(response.json())