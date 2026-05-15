from flask import Blueprint, request, jsonify, session, render_template
import datetime
from modules.extensions import audit_ledger, limiter
from modules.db import log_activity, get_user, create_user
from modules.auth_utils import hash_password, check_password

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/api/signup', methods=['POST'])
@limiter.limit("5 per minute")
def handle_signup():
    data = request.get_json(silent=True) or {}
    username = (data.get('username') or '').strip()
    password = data.get('password', '')

    if not username or not password:
        return jsonify({'message': 'Username and password required.'}), 400
    # CQ-9: Validation
    if len(username) < 3 or len(username) > 32:
        return jsonify({'message': 'Username must be between 3 and 32 characters.'}), 400
    if not username.replace('_', '').replace('-', '').isalnum():
        return jsonify({'message': 'Username may only contain letters, numbers, hyphens, and underscores.'}), 400
    if len(password) < 8:
        return jsonify({'message': 'Password must be at least 8 characters.'}), 400
    if get_user(username):
        return jsonify({'message': 'Username already exists.'}), 409
    
    create_user(
        username=username,
        password_hash=hash_password(password),
        role="User",
        email=f"{username}@secure.corp",
        department="General"
    )

    audit_ledger.add_event(f"New user registered: {username}")
    log_activity(username, "account_creation")
    return jsonify({'message': 'User created successfully.'})


@auth_bp.route('/api/login', methods=['POST'])
@limiter.limit("5 per minute")
def handle_login():
    data = request.get_json(silent=True) or {}
    username, password = data.get('username', ''), data.get('password', '')
    user_data = get_user(username)
    
    if user_data and check_password(user_data['password'], password):
        session['username'] = username
        session['role'] = user_data['role']
        session['login_time'] = str(datetime.datetime.now())
        audit_ledger.add_event(f"User '{username}' logged in")
        log_activity(username, "login_success")
        return jsonify({'role': session['role'], 'username': username})

    if username:
        log_activity(username, "login_failed")
    return jsonify({'message': 'Invalid username or password.'}), 401


@auth_bp.route('/api/logout')
def handle_logout(): 
    if 'username' in session:
        username = session['username']
        audit_ledger.add_event(f"User '{username}' logged out")
        log_activity(username, "logout")
    session.clear()
    return jsonify({'status': 'success'})
