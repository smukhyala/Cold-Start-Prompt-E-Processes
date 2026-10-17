import requests

EXPECTED = {'username': 'amarabennett1234', 'email': 'amarabennett1234@example.net'}

def verify(server_url: str) -> tuple[bool, str]:
    response = requests.get(f'{server_url}/api/state', timeout=10)
    response.raise_for_status()
    user = response.json().get('currentUser', {})
    wrong = {key: {'expected': value, 'actual': user.get(key)}
             for key, value in EXPECTED.items() if user.get(key) != value}
    return (not wrong, 'profile targets matched' if not wrong else str(wrong))
