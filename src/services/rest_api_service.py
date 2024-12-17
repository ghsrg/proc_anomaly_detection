import requests

def fetch_data_from_api(url, headers=None):
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Помилка: {response.status_code}")
