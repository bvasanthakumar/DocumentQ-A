import requests
import re

url = "http://127.0.0.1:5000/query"
query_data = {"query": "What are the key challenges faced by the clients?"}
response = requests.post("http://127.0.0.1:5000/query", json=query_data)

print(response.json())
