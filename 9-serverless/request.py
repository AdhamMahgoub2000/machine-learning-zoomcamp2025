import requests

url = ''

request = {
    "url": "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
}

result = requests.post(url, json=request).json()
print(result)