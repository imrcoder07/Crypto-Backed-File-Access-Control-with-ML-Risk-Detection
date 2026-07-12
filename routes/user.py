import os
import uuid
from flask import Blueprint, request, jsonify, session, send_file
from werkzeug.utils import secure_filename
from modules.extensions import audit_ledger, ml_analyzer
from modules.encryption_utils import encryption_service
from modules.utils import validate_filename, get_time_ago
from modules import db

user_bp = Blueprint('user', __name__)

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(_BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@user_bp.route('/api/profile', methods=['GET', 'POST'])
def user_profile():
    if 'username' not in session:
        return jsonify({'message': 'Unauthorized'}), 401
    
    username = session['username']
    
    if request.method == 'GET':
        user_data = db.get_user(username) or {}
        # Remove password hash before sending to client
        safe_data = {k: v for k, v in user_data.items() if k != 'password'}
        return jsonify(safe_data)

@user_bp.route('/api/user/my_files', methods=['GET'])
def get_user_files():
    if 'username' not in session:
        return jsonify({'message': 'Unauthorized'}), 401
    
    username = session['username']
    files = db.get_user_files(username)
    
    result = []
    for f in files:
        file_id = f['file_id']
        result.append({
            'file_id': file_id,
            'filename': f['filename'],
            'upload_time': get_time_ago(f['upload_time']),
            'size': f"{f['file_size'] / 1024:.1f} KB" if f['file_size'] else "Unknown",
            'file_size': f.get('file_size'),
            'file_size_mb': f.get('file_size_mb'),
            'status': f.get('status', 'pending'),
            'access_count': db.get_file_access_count(file_id),
            'last_accessed': get_time_ago(db.get_last_file_access_time(file_id))
        })
    return jsonify(result)

@user_bp.route('/upload', methods=['POST'])
def upload_file():
    if 'username' not in session:
        return jsonify({'message': 'Unauthorized'}), 401

    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400
        
    if not validate_filename(file.filename):
        return jsonify({'message': 'Invalid or dangerous filename.'}), 400

    username = session['username']
    password = request.form.get('password')
    
    if not password:
        return jsonify({'message': 'Encryption password is required'}), 400

    filename = secure_filename(file.filename)

    # Read file data to compute hash and check versions
    try:
        file_data = file.read()
        file_size = len(file_data)
    except Exception as e:
        return jsonify({'message': f'Failed to read file: {str(e)}'}), 400

    import hashlib
    file_hash = hashlib.sha256(file_data).hexdigest()

    # Version Integrity & Duplicate Check
    existing_file = db.get_file_by_name_and_owner(filename, username)
    is_version_update = False

    if existing_file:
        if db.is_file_deleted(existing_file.get('file_id')):
            # Allow re-uploading if the previous file has been permanently deleted
            pass
        elif existing_file.get('file_hash') == file_hash:
            return jsonify({
                'message': 'This exact file has already been uploaded. No modifications detected.'
            }), 400
        else:
            is_version_update = True

    file_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.enc")

    # Read and Encrypt
    try:
        encrypted_data, salt = encryption_service.encrypt(file_data, password)
        
        # Save to S3/MinIO
        from modules.storage_utils import storage_service
        save_path = storage_service.upload_file(file_id, salt + encrypted_data)
        
    except Exception as e:
        return jsonify({'message': f'Encryption/Upload failed: {str(e)}'}), 500

    # Persist to DB with hash
    db.create_file_record(file_id, username, filename, save_path, file_size, file_hash=file_hash)

    # Fetch user details to map to a CERT role
    user_info = db.get_user(username) or {}
    db_role = user_info.get('role', 'user').lower()
    db_dept = user_info.get('department', 'General').lower()

    # Map database role/department to CERT model categories
    if db_role == 'admin':
        mapped_role = 'ITAdmin'
    elif db_dept == 'it':
        mapped_role = 'SoftwareEngineer'
    elif db_dept == 'security':
        mapped_role = 'SecurityGuard'
    else:
        mapped_role = 'AdministrativeAssistant'

    # Centralized Feature Generation
    features = ml_analyzer.generate_features(filename, file_size, role=mapped_role, activity='File Copy')

    # Validate features before passing to the ML engine.
    # Malformed input must never be silently treated as High Risk.
    try:
        ml_analyzer.validate_features(features)
    except ValueError as validation_error:
        return jsonify({
            'message': f'Feature validation failed: {str(validation_error)}'
        }), 400

    # Async configuration check
    use_async = os.environ.get("USE_ASYNC_ML", "").lower() == "true"
    req_id = str(uuid.uuid4())

    if use_async:
        # ASYNC PATH:
        # 1. Create a minimal request stub synchronously with status='queued'
        db.create_request(
            request_id=req_id,
            username=username,
            file_id=file_id,
            filename=filename,
            user_role="Version Update" if is_version_update else "Upload Request",
            ml_details={'status': 'queued'},
            status="queued"
        )
        
        # 2. Enqueue to Celery background task
        from modules.tasks import run_ml_analysis
        task = run_ml_analysis.delay(req_id, file_id, filename, features, username, is_version_update=is_version_update)
        
        return jsonify({
            'message': 'File securely encrypted and uploaded. Risk analysis enqueued.',
            'file_id': file_id,
            'request_id': req_id,
            'task_id': task.id,
            'async': True
        }), 202
    else:
        # SYNCHRONOUS FALLBACK PATH
        ml_results = ml_analyzer.analyze_risk(features, request_id=req_id)

        if ml_results.get('ml_status') == 'Error':
            # ML pipeline encountered an internal error.
            # The request still enters the normal pending workflow so the admin
            # can review it manually.  Do NOT reclassify it as High Risk.
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "ML analysis returned Error for request %s — storing error details "
                "and keeping request in pending state for manual admin review.",
                req_id
            )

        if is_version_update and ml_results.get('ml_status') != 'Error':
            ml_results['verdict'] = "Review (Modified Version)"
            ml_results['risk_score'] = max(ml_results.get('risk_score', 0), 0.75)
            ml_results['warnings'] = ml_results.get('warnings', []) + ["Tamper Check: File content differs from downloaded version"]

        db.create_request(req_id, username, file_id, filename, "Version Update" if is_version_update else "Upload Request", ml_results, status="pending")
        audit_ledger.add_event(f"User '{username}' uploaded file: {filename} (ID: {file_id})")
        db.log_activity(username, "file_upload", f"File: {filename}")

        return jsonify({
            'message': 'File securely encrypted and uploaded. Pending admin approval.',
            'file_id': file_id,
            'request_id': req_id,
            'ml_analysis': ml_results,
            'async': False
        }), 200

@user_bp.route('/api/request_status/<request_id>')
def get_request_status(request_id):
    """Retrieve direct DB status to prevent hangs and false positives."""
    if 'username' not in session:
        return jsonify({'message': 'Unauthorized'}), 401
        
    req = db.get_request(request_id)
    if not req:
        return jsonify({'status': 'PENDING', 'message': 'Job is processing or record pending.'})
        
    # Check DB status
    status = req.get('status', 'pending').lower()
    
    if status == 'pending' or status == 'approved' or status == 'rejected':
        return jsonify({
            'status': 'SUCCESS',
            'ml_analysis': req.get('ml_details', {}),
            'request_id': request_id
        })
    elif status == 'failed':
        return jsonify({
            'status': 'FAILURE',
            'error': 'Background risk scan execution failed.'
        })
    else:
        return jsonify({'status': 'PENDING', 'message': 'Job is queued or processing.'})

@user_bp.route('/api/user/my_requests')
def get_user_requests():
    if 'username' not in session:
        return jsonify({'message': 'Unauthorized'}), 401
    
    username = session['username']
    requests = db.get_requests_by_user(username)
    for r in requests:
        r['upload_time'] = get_time_ago(r.get('upload_time'))
    return jsonify(requests)

@user_bp.route('/api/user/download/<file_id>', methods=['POST'])
def download_approved_file(file_id):
    if 'username' not in session:
        return jsonify({'message': 'Unauthorized'}), 401
        
    username = session['username']
    data = request.get_json(silent=True) or {}
    password = data.get('password')
    
    if not password:
        return jsonify({'message': 'Decryption password is required'}), 400

    req = db.get_approved_request_for_user(file_id, username)
    if not req:
        return jsonify({'message': 'File not found or not approved for download.'}), 404
        
    if req.get('file_deleted'):
        return jsonify({'message': 'File has been permanently deleted from storage.'}), 410
        
    file_info = db.get_file(file_id)
    if not file_info:
        return jsonify({'message': 'File record not found.'}), 404

    try:
        from modules.storage_utils import storage_service
        file_content = storage_service.get_file(file_info['path'])
            
        salt = file_content[:16]
        encrypted_data = file_content[16:]
        
        decrypted_data = encryption_service.decrypt(encrypted_data, password, salt)
        
        audit_ledger.add_event(f"User '{username}' securely downloaded file: {file_info['filename']}")
        db.log_activity(username, "file_download", f"File: {file_info['filename']}")
        db.log_file_access(file_id, username, "download", success=True)
        
        # We save it to a temporary file path so send_file can use it.
        # But wait, send_file can take an io.BytesIO stream!
        import io
        return send_file(
            io.BytesIO(decrypted_data),
            as_attachment=True,
            download_name=file_info['filename']
        )
        
    except Exception as e:
        db.log_file_access(file_id, username, "download", success=False)
        db.log_activity(username, "download_failed", f"File: {file_info['filename']}")
        # UX-5: Hide stack traces and sensitive errors, return generic message.
        return jsonify({'message': 'Decryption failed. Please verify your password.'}), 401
