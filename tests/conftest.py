import pytest
from app import app
from modules import db

@pytest.fixture
def client():
    app.config['TESTING'] = True
    # We might want an in-memory db or a test DB in a real scenario
    # For now, we will test the current DB or mock
    with app.test_client() as client:
        with app.app_context():
            yield client
