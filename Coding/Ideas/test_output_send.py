import requests

_id = "your algo name before the .log"
id_token = " get from cookies as .idToken"
headers = {
    'Authorization': "Bearer "+id_token
}

response = requests.get(
    url=f"https://bz97lt8b1e.execute-api.eu-west-1.amazonaws.com/prod/results/tutorial/{_id}",
     headers=headers
)

if __name__ == '__main__':
    b = response.json()