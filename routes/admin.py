import os
import io
import json
import logging
import datetime
from flask import Blueprint, request, jsonify, session, send_file
from modules.extensions import audit_ledger, ml_analyzer
from modules.audit_utils import log_event
from modules.db import get_all_users
from modules.utils import get_time_ago
from modules import db
from modules.storage_utils import storage_service

logger = logging.getLogger(__name__)

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
        
    return jsonify({
        'total_blocks': len(chain),
        'last_update': chain[-1]['timestamp'] if chain else None,
        'chain': chain
    })

@admin_bp.route('/api/admin/blockchain_log')
def get_blockchain_log():
    """Compatibility alias for the audit log endpoint."""
    return get_audit_log()

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
        # NOTE: approved_at is intentionally kept as a raw ISO timestamp string.
        # The frontend uses new Date(file.approved_at).toLocaleString() to format it.
        # Do NOT transform it with get_time_ago() — that produces "X days ago" which
        # new Date() cannot parse, causing "Invalid Date" in the UI.
        req['upload_time'] = get_time_ago(req.get('upload_time'))
        
        file_id = req.get('file_id')
        req['access_count'] = db.get_file_access_count(file_id) if file_id else 0
        last_accessed = db.get_last_file_access_time(file_id) if file_id else None
        req['last_accessed'] = get_time_ago(last_accessed)
        
    return jsonify(requests)

@admin_bp.route('/api/admin/download_file/<file_id>', methods=['POST'])
def admin_download_file(file_id):
    """Download the raw encrypted blob of any file for admin audit purposes.
    
    Fetches the encrypted payload from S3/MinIO (not local disk) and streams
    it back to the admin browser as a binary attachment. The admin receives
    the encrypted ciphertext — they cannot decrypt it without the user's password.
    This is intentional: zero-knowledge audit trail.
    """
    if not is_admin():
        return jsonify({'message': 'Unauthorized'}), 401

    file_info = db.get_file(file_id)
    if not file_info:
        return jsonify({'message': 'File not found'}), 404

    # Prevent download if file is deleted
    with db.get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT file_deleted FROM requests WHERE file_id = %s;", (file_id,))
            row = cur.fetchone()
            if row and row[0]:
                return jsonify({'message': 'File has been permanently deleted from storage.'}), 410

    admin_user = session['username']
    object_path = file_info.get('path', '')

    try:
        # Fetch encrypted bytes from S3/MinIO — no local disk access
        encrypted_bytes = storage_service.get_file(object_path)

        log_event(
            action="ADMIN_FILE_DOWNLOAD",
            admin=admin_user,
            file_id=file_id,
            filename=file_info['filename'],
            details=f"Admin '{admin_user}' downloaded encrypted blob: {file_info['filename']} (id={file_id})"
        )
        db.log_activity(admin_user, "admin_download", f"Encrypted blob: {file_info['filename']}")
        db.log_file_access(file_id, admin_user, "admin_audit_download", success=True)

        return send_file(
            io.BytesIO(encrypted_bytes),
            as_attachment=True,
            download_name=f"encrypted_{file_info['filename']}",
            mimetype='application/octet-stream'
        )
    except Exception as e:
        logger.error(f"Admin download failed for file_id={file_id}: {e}", exc_info=True)
        db.log_file_access(file_id, admin_user, "admin_audit_download", success=False)
        return jsonify({'message': 'Download failed. Could not retrieve file from storage.'}), 500

@admin_bp.route('/api/admin/approve_request', methods=['POST'])
def approve_request():
    if not is_admin():
        return jsonify({'message': 'Unauthorized'}), 401
        
    data = request.get_json(silent=True) or {}
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
            
        log_event(
            action="ACCESS_REQUEST_APPROVED",
            admin=admin_user,
            request_id=request_id,
            details=f"Admin '{admin_user}' APPROVED access request {request_id}"
        )
        db.log_activity(admin_user, "approve_request", f"Request ID: {request_id}")
        
        return jsonify({'message': 'Request approved successfully'})
    except Exception as e:
        return jsonify({'message': f'Approval failed: {str(e)}'}), 500

@admin_bp.route('/api/admin/reject_request', methods=['POST'])
def reject_request():
    if not is_admin():
        return jsonify({'message': 'Unauthorized'}), 401
        
    data = request.get_json(silent=True) or {}
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
            
        log_event(
            action="ACCESS_REQUEST_REJECTED",
            admin=admin_user,
            request_id=request_id,
            details=f"Admin '{admin_user}' REJECTED access request {request_id}"
        )
        db.log_activity(admin_user, "reject_request", f"Request ID: {request_id}")
        
        return jsonify({'message': 'Request rejected successfully'})
    except Exception as e:
        return jsonify({'message': f'Rejection failed: {str(e)}'}), 500


@admin_bp.route('/api/admin/rejected_requests')
def get_rejected_requests():
    if not is_admin():
        return jsonify({'message': 'Unauthorized'}), 401
        
    requests = db.get_rejected_requests()
    for req in requests:
        req['upload_time'] = get_time_ago(req.get('upload_time'))
    return jsonify(requests)


@admin_bp.route('/api/admin/request/<request_id>/delete-file', methods=['DELETE'])
def delete_request_file(request_id):
    if 'username' not in session:
        return jsonify({'message': 'Unauthorized'}), 401
    if not is_admin():
        return jsonify({'message': 'Forbidden'}), 403
        
    admin_user = session['username']
    try:
        success, reason = db.delete_encrypted_file(request_id, admin_user)
        if reason == "Already deleted":
            return jsonify({'message': 'File already removed from storage.'}), 200
            
        if not success:
            if reason == "Missing from storage":
                req = db.get_request(request_id)
                filename = req.get('filename') if req else 'Unknown'
                log_event(
                    action="STORAGE_INCONSISTENCY",
                    admin=admin_user,
                    request_id=request_id,
                    filename=filename,
                    details="Encrypted file missing from storage during delete operation",
                    metadata={'reason': 'Encrypted file missing from storage during delete operation'}
                )
                db.log_activity(admin_user, "storage_inconsistency_alert", f"Missing file request ID: {request_id}")
                return jsonify({'message': 'Storage inconsistency detected: Encrypted file is missing. Administrator attention required.'}), 409
            else:
                return jsonify({'message': f'Deletion failed: {reason}'}), 400
                
        req = db.get_request(request_id)
        filename = req.get('filename') if req else 'Unknown'
        status = req.get('status') if req else 'Unknown'
        log_event(
            action="FILE_DELETED",
            admin=admin_user,
            request_id=request_id,
            filename=filename,
            details=f"Admin '{admin_user}' deleted file: {filename}",
            metadata={'status': status, 'reason': 'Storage Cleanup'}
        )
        db.log_activity(admin_user, "file_deleted", f"Deleted file for request ID: {request_id}")
        
        return jsonify({'message': 'Encrypted file deleted successfully.'}), 200
    except Exception as e:
        logger.error(f"Error during file deletion for request_id={request_id}: {e}", exc_info=True)
        return jsonify({'message': f'Server error during deletion: {str(e)}'}), 500


@admin_bp.route('/api/admin/user_activity', methods=['GET'])
def get_admin_user_activity():
    if not is_admin():
        return jsonify({'message': 'Unauthorized'}), 401

    admin_user = session['username']
    username = request.args.get('username', '').strip()
    if not username:
        return jsonify({'message': 'Username query parameter is required.'}), 400

    # User existence verification
    user_info = db.get_user(username)
    if not user_info:
        return jsonify({'message': f"User '{username}' not found."}), 404

    # Pagination parsing
    try:
        limit = int(request.args.get('limit', 50))
    except (TypeError, ValueError):
        limit = 50
    limit = max(1, min(limit, 100))

    try:
        offset = int(request.args.get('offset', 0))
    except (TypeError, ValueError):
        offset = 0
    offset = max(0, offset)

    # Retrieval via extended DB helper
    try:
        rows, total = db.get_user_activity(username, count=limit, offset=offset, include_total=True)
    except Exception as e:
        logger.error(f"Failed to load user activity for {username}: {e}", exc_info=True)
        return jsonify({'message': 'Internal error retrieving user activity.'}), 500

    # Format structured output only (presentation string formatting deferred to UI)
    activities_list = []
    for row in rows:
        ts = row.get('ts')
        ts_str = ts.strftime('%Y-%m-%d %H:%M:%S') if hasattr(ts, 'strftime') else str(ts)
        activities_list.append({
            'activity': row.get('activity', ''),
            'details': row.get('details') or '',
            'timestamp': ts_str
        })

    # Log standardized admin view event (Phase 3 audit standardization compliant)
    log_event(
        action="ADMIN_USER_ACTIVITY_VIEW",
        admin=admin_user,
        username=username,
        details=f"Viewed user activity history for user: {username}"
    )

    return jsonify({
        'username': username,
        'activities': activities_list,
        'pagination': {
            'limit': limit,
            'offset': offset,
            'total': total,
            'has_more': (offset + len(rows)) < total
        }
    }), 200


@admin_bp.route('/api/admin/audit', methods=['GET'])
def get_admin_audit():
    if not is_admin():
        return jsonify({'message': 'Unauthorized'}), 401

    # Extract filter parameters from query parameters
    query = request.args.get('query')
    username = request.args.get('username')
    admin = request.args.get('admin')
    action = request.args.get('action')
    severity = request.args.get('severity')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    limit = request.args.get('limit')
    offset = request.args.get('offset')

    from modules.audit_query_service import AuditQueryFilter, AuditQueryService

    # 1. Instantiate validation filter object
    query_filter = AuditQueryFilter(
        query=query,
        username=username,
        admin=admin,
        action=action,
        severity=severity,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset
    )

    # 2. Delegate query execution to decoupled AuditQueryService
    try:
        results = AuditQueryService.query_audit_logs(query_filter)
        return jsonify(results), 200
    except Exception as e:
        logger.error(f"Audit log search failed: {e}", exc_info=True)
        return jsonify({'message': 'Internal search query execution failed.'}), 500



