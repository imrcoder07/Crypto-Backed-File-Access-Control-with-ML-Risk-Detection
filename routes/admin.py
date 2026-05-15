import os
import datetime
from flask import Blueprint, request, jsonify, session, send_file
import io
from modules.extensions import audit_ledger, ml_analyzer
from modules.db import get_all_users
from modules.utils import get_time_ago
from modules import db

admin_bp = Blueprint('admin', __name__)

def is_admin():
    return session.get('role') == 'Admin'

@admin_bp.route('/api/admin/users')
def get_admin_users():
    if not is_admin():
        return jsonify({'message': 'Unauthorized'}), 401
    
    users_data = []
    for data in get_all_users():
        username = data['username']
        user_info = {
            'username': username,
            'role': data['role'],
            'department': data.get('department', 'Unknown')
        }
        
        # Get activity from DB
        recent_activity = db.get_user_activity(username, count=1)
        if recent_activity:
            user_info['last_active'] = get_time_ago(recent_activity[0]['ts'])
        else:
            user_info['last_active'] = "Never"
            
        users_data.append(user_info)
        
    return jsonify(users_data)

@admin_bp.route('/api/admin/audit_log')
def get_audit_log():
    if not is_admin():
        return jsonify({'message': 'Unauthorized'}), 401
    
    # We return the formatted chain snapshot
    chain = audit_ledger.get_chain_snapshot()
    for block in chain:
        block['timestamp_formatted'] = get_time_ago(block['timestamp'])
    return jsonify(chain)

@admin_bp.route('/api/admin/security_alerts')
def get_security_alerts():
    if not is_admin():
        return jsonify({'message': 'Unauthorized'}), 401
    
    # Get ML alerts, and we can also add some dummy logic or db logic if we want
    alerts = list(ml_analyzer.security_alerts)
    for alert in alerts:
        alert['time_ago'] = get_time_ago(alert['timestamp'])
    return jsonify(alerts)

@admin_bp.route('/api/admin/pending_requests')
def get_pending_requests():
    if not is_admin():
        return jsonify({'message': 'Unauthorized'}), 401
        
    requests = db.get_pending_requests()
    for req in requests:
        req['upload_time'] = get_time_ago(req['upload_time'])
    return jsonify(requests)

@admin_bp.route('/api/admin/approved_files')
def get_approved_files():
    if not is_admin():
        return jsonify({'message': 'Unauthorized'}), 401
        
    requests = db.get_approved_requests()
    for req in requests:
        req['approved_at'] = get_time_ago(req.get('approved_at'))
        req['upload_time'] = get_time_ago(req.get('upload_time'))
        
        file_id = req.get('file_id')
        req['access_count'] = db.get_file_access_count(file_id) if file_id else 0
        last_accessed = db.get_last_file_access_time(file_id) if file_id else None
        req['last_accessed'] = get_time_ago(last_accessed)
        
    return jsonify(requests)

@admin_bp.route('/api/admin/download_file/<file_id>', methods=['POST'])
def admin_download_file(file_id):
    if not is_admin():
        return jsonify({'message': 'Unauthorized'}), 401
        
    # Admin can download any approved file, but they still need the original password,
    # OR we are just allowing them to download the *encrypted* version?
    # For auditing, let's say they just download the raw encrypted blob.
    file_info = db.get_file(file_id)
    if not file_info:
        return jsonify({'message': 'File not found'}), 404
        
    admin_user = session['username']
    
    try:
        audit_ledger.add_event(f"Admin '{admin_user}' downloaded encrypted file blob: {file_info['filename']}")
        db.log_activity(admin_user, "admin_download", f"Encrypted blob: {file_info['filename']}")
        db.log_file_access(file_id, admin_user, "admin_audit_download", success=True)
        
        return send_file(
            file_info['path'],
            as_attachment=True,
            download_name=f"encrypted_{file_info['filename']}"
        )
    except Exception as e:
        db.log_file_access(file_id, admin_user, "admin_audit_download", success=False)
        return jsonify({'message': f'Download failed: {str(e)}'}), 500

@admin_bp.route('/api/admin/approve_request', methods=['POST'])
def approve_request():
    if not is_admin():
        return jsonify({'message': 'Unauthorized'}), 401
        
    data = request.json
    request_id = data.get('request_id')
    notes = data.get('notes', '')
    admin_user = session['username']
    
    if not request_id:
        return jsonify({'message': 'Request ID missing'}), 400

    approved_at = str(datetime.datetime.now())
    try:
        updated_req = db.approve_request(request_id, admin_user, notes, approved_at)
        if not updated_req:
            return jsonify({'message': 'Request not found or already processed'}), 404
            
        audit_ledger.add_event(f"Admin '{admin_user}' APPROVED access request {request_id}")
        db.log_activity(admin_user, "approve_request", f"Request ID: {request_id}")
        
        return jsonify({'message': 'Request approved successfully'})
    except Exception as e:
        return jsonify({'message': f'Approval failed: {str(e)}'}), 500

@admin_bp.route('/api/admin/reject_request', methods=['POST'])
def reject_request():
    if not is_admin():
        return jsonify({'message': 'Unauthorized'}), 401
        
    data = request.json
    request_id = data.get('request_id')
    notes = data.get('notes', '')
    admin_user = session['username']
    
    if not request_id:
        return jsonify({'message': 'Request ID missing'}), 400

    rejected_at = str(datetime.datetime.now())
    try:
        updated_req = db.reject_request(request_id, admin_user, notes, rejected_at)
        if not updated_req:
            return jsonify({'message': 'Request not found or already processed'}), 404
            
        audit_ledger.add_event(f"Admin '{admin_user}' REJECTED access request {request_id}")
        db.log_activity(admin_user, "reject_request", f"Request ID: {request_id}")
        
        return jsonify({'message': 'Request rejected successfully'})
    except Exception as e:
        return jsonify({'message': f'Rejection failed: {str(e)}'}), 500
