import json

import json

def test_signup_api(client):
    response = client.post('/api/signup', json={
        'username': 'testuser123',
        'password': 'password1234'
    })
    assert response.status_code in [200, 409] # 200 on success, 409 if exists
    
    data = response.get_json()
    assert 'message' in data

def test_invalid_login_api(client):
    response = client.post('/api/login', json={
        'username': 'nonexistentuser',
        'password': 'wrongpassword'
    })
    
    assert response.status_code == 401
    data = response.get_json()
    assert data['message'] == 'Invalid username or password.'
