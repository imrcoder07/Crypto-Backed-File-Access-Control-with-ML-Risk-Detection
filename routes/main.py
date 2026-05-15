from flask import Blueprint, render_template, jsonify
from modules.extensions import audit_ledger, ml_analyzer
from modules.db import get_all_users
from modules.db import (
    all_requests_ml_details,
    approved_count,
    database_status,
    file_count,
    password_protected_count,
    pending_count,
)

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index(): 
    return render_template('index.html')

@main_bp.route('/api/dashboard_data')
def get_dashboard_data():
    req_details = all_requests_ml_details()
    avg_risk = sum(r.get('risk_score', 0) for r in req_details) / len(req_details) if req_details else 0.0
    
    return jsonify({
        'pending_requests': pending_count(),
        'approved_files': approved_count(),
        'avg_risk': round(avg_risk, 2),
        'chain_length': audit_ledger.chain_length
    })

@main_bp.route('/api/system_stats')
def get_system_stats():
    # CQ-5: Actually report real ML status instead of static True
    req_details = all_requests_ml_details()
    allowed = sum(1 for item in req_details if "allow" in str(item).lower())
    denied = sum(1 for item in req_details if "deny" in str(item).lower())
    review_required = max(len(req_details) - allowed - denied, 0)

    return jsonify({
        'ml_active': ml_analyzer.models_loaded,
        'blockchain_active': True,
        'chain_length': audit_ledger.chain_length,
        'blockchain_events': audit_ledger.chain_length,
        'database': database_status(),
        'database_upgraded': True,
        'database_engine': 'PostgreSQL',
        'total_files': file_count(),
        'pending_requests': pending_count(),
        'approved_requests': approved_count(),
        'password_protected_files': password_protected_count(),
        'total_users': len(get_all_users()),
        'ml_analysis': {
            'total_analyzed': len(req_details),
            'allowed': allowed,
            'denied': denied,
            'review_required': review_required,
        },
    })

@main_bp.route('/api/health')
def health_check():
    return jsonify({'status': 'healthy', 'db': 'PostgreSQL connected', 'ml_loaded': ml_analyzer.models_loaded})
