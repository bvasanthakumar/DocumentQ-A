import requests

url = "http://127.0.0.1:5000/upload"
files = {"file": open("Data-Modernization-Cloud-data-foundation.pdf", "rb")}
response = requests.post(url, files=files)
print(response.json())
