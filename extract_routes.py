import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

def extract_funcs(funcs, bp_name, additional_imports=''):
    output = []
    output.append("from flask import Blueprint, request, jsonify, session, send_file, render_template")
    output.append("import os, json, datetime, time")
    output.append("from werkzeug.utils import secure_filename")
    output.append("import io")
    output.append("from modules import db")
    output.append("from modules.utils import get_time_ago, validate_filename")
    output.append("from modules.extensions import blockchain, ml_analyzer")
    output.append("from modules.encryption_utils import AdvancedEncryption")
    output.append("import bcrypt")
    output.append(additional_imports)
    output.append(f"\n{bp_name}_bp = Blueprint('{bp_name}', __name__)\n")
    
    for func in funcs:
        pattern = r'(@app\.route\([\s\S]*?def ' + func + r'\(.*?\):[\s\S]*?)(?=\n@app\.route|\n# =|\nif __name__ ==|$)'
        match = re.search(pattern, content)
        if match:
            func_code = match.group(1)
            func_code = func_code.replace('@app.route', f'@{bp_name}_bp.route')
            output.append(func_code)
            
    with open(f'routes/{bp_name}.py', 'w', encoding='utf-8') as f:
        f.write('\n'.join(output))
    print(f'Created routes/{bp_name}.py')

auth_funcs = ['handle_signup', 'handle_login', 'handle_logout']
extract_funcs(auth_funcs, 'auth')

main_funcs = ['index', 'get_dashboard_data', 'get_system_stats', 'health_check']
extract_funcs(main_funcs, 'main')

user_funcs = ['user_profile', 'upload_file', 'get_user_requests', 'download_approved_file', 'get_user_files']
extract_funcs(user_funcs, 'user')

admin_funcs = ['get_admin_users', 'get_blockchain_log', 'get_security_alerts', 'get_pending_requests', 'get_approved_files', 'admin_download_file', 'approve_request', 'reject_request']
extract_funcs(admin_funcs, 'admin')
