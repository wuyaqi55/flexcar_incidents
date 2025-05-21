import requests

# Replace with the API endpoint you want to call
url = "https://jsonplaceholder.typicode.com/users"

response = requests.get(url)

if response.status_code == 200:
    data = response.json()  # Parse JSON response
    print(data)
else:
    print(f"Request failed with status code {response.status_code}")