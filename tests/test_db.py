import pytest
from modules import db

def test_db_get_user(client):
    # This should at least run without crashing
    user = db.get_user("admin")
    if user:
        assert user['username'] == "admin"

def test_db_get_all_users(client):
    users = db.get_all_users()
    assert isinstance(users, list)
