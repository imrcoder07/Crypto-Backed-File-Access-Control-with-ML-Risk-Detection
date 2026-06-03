import os
import uuid
import pytest
from unittest.mock import patch, MagicMock
from modules import db
from modules.tasks import run_ml_analysis

@pytest.fixture
def logged_in_client(client):
    # Register and log in a test user
    username = f"testuser_{uuid.uuid4().hex[:6]}"
    password = "password1234"
    db.create_user(username, password, "user", f"{username}@example.com", "IT")
    
    with client.session_transaction() as sess:
        sess['username'] = username
    
    yield client, username

@patch('modules.storage_utils.storage_service.upload_file')
def test_upload_file_sync_mode(mock_upload, logged_in_client):
    client, username = logged_in_client
    mock_upload.return_value = "fake_s3_path"
    
    # Ensure sync mode is active
    os.environ['USE_ASYNC_ML'] = 'false'
    
    import io
    data = {
        'file': (io.BytesIO(b"test file content"), 'testfile.txt'),
        'password': 'encryption_password'
    }
    
    response = client.post('/upload', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    
    res_data = response.get_json()
    assert res_data['async'] is False
    assert 'request_id' in res_data
    
    # Verify in DB
    req = db.get_request(res_data['request_id'])
    assert req is not None
    assert req['status'] == 'pending'  # Sync mode directly creates pending review requests
    assert req['filename'] == 'testfile.txt'

@patch('modules.storage_utils.storage_service.upload_file')
@patch('modules.tasks.run_ml_analysis.delay')
def test_upload_file_async_mode(mock_delay, mock_upload, logged_in_client):
    client, username = logged_in_client
    mock_upload.return_value = "fake_s3_path"
    
    # Mock celery task.delay() call
    mock_task = MagicMock()
    mock_task.id = "mock-celery-task-id-12345"
    mock_delay.return_value = mock_task
    
    # Enable async mode
    os.environ['USE_ASYNC_ML'] = 'true'
    
    import io
    data = {
        'file': (io.BytesIO(b"test file content async"), 'asyncfile.txt'),
        'password': 'encryption_password'
    }
    
    response = client.post('/upload', data=data, content_type='multipart/form-data')
    assert response.status_code == 202
    
    res_data = response.get_json()
    assert res_data['async'] is True
    assert res_data['task_id'] == "mock-celery-task-id-12345"
    
    # Verify stub created in DB with status 'queued'
    req = db.get_request(res_data['request_id'])
    assert req is not None
    assert req['status'] == 'queued'
    assert req['filename'] == 'asyncfile.txt'

def test_request_status_polling_api(logged_in_client):
    client, username = logged_in_client
    req_id = f"req_{uuid.uuid4().hex}"
    file_id = f"file_{uuid.uuid4().hex}"
    
    # 1. Create file record to satisfy foreign key constraint
    db.create_file_record(file_id, username, "test_poll.txt", "fake_path", 100)
    
    # 2. Create a stub manually with status='queued'
    db.create_request(
        request_id=req_id,
        username=username,
        file_id=file_id,
        filename="test_poll.txt",
        user_role="Upload Request",
        ml_details={'status': 'queued'},
        status="queued"
    )
    
    # Poll and check PENDING status mapping
    response = client.get(f'/api/request_status/{req_id}')
    assert response.status_code == 200
    assert response.get_json()['status'] == 'PENDING'
    
    # 2. Update status to 'processing' and check PENDING status mapping
    db.update_request_status_only(req_id, 'processing')
    response = client.get(f'/api/request_status/{req_id}')
    assert response.status_code == 200
    assert response.get_json()['status'] == 'PENDING'
    
    # 3. Update status to 'pending' (completed risk analysis) and check SUCCESS mapping
    db.update_request_ml_results(
        request_id=req_id,
        ml_verdict="Allow",
        ml_details={'risk_score': 0.15, 'is_risky': False},
        new_status="pending"
    )
    response = client.get(f'/api/request_status/{req_id}')
    assert response.status_code == 200
    res_json = response.get_json()
    assert res_json['status'] == 'SUCCESS'
    assert res_json['ml_analysis']['risk_score'] == 0.15
    
    # 4. Update status to 'failed' (Celery retry exhausted) and check FAILURE mapping
    db.update_request_status_only(req_id, 'failed')
    response = client.get(f'/api/request_status/{req_id}')
    assert response.status_code == 200
    assert response.get_json()['status'] == 'FAILURE'

@patch('modules.extensions.audit_ledger.add_event')
def test_celery_task_execution(mock_ledger, logged_in_client):
    client, username = logged_in_client
    req_id = f"req_{uuid.uuid4().hex}"
    file_id = f"file_{uuid.uuid4().hex}"
    
    # Save a file metadata record first (so referential integrity with files table doesn't fail)
    db.create_file_record(file_id, username, "task_file.txt", "fake_path", 100)
    
    # Create request stub with 'queued' status
    db.create_request(
        request_id=req_id,
        username=username,
        file_id=file_id,
        filename="task_file.txt",
        user_role="Upload Request",
        ml_details={'status': 'queued'},
        status="queued"
    )
    
    features = {
        'file_size': 100,
        'name_length': len("task_file.txt"),
        'is_executable': 0,
        'has_special_chars': 0,
        'upload_hour': 12,
        'user_trust_score': 0.85
    }
    
    # Run the Celery task synchronously in the current thread by calling .run() or calling task directly
    run_ml_analysis(req_id, file_id, "task_file.txt", features, username)
    
    # Check that status was updated to 'pending'
    req = db.get_request(req_id)
    assert req is not None
    assert req['status'] == 'pending'
    assert req['ml_verdict'] is not None
    assert mock_ledger.called
