import logging
import json
from modules.celery_app import celery_instance
from modules.extensions import ml_analyzer, audit_ledger
from modules import db

logger = logging.getLogger(__name__)

@celery_instance.task(name="tasks.run_ml_analysis", bind=True, max_retries=3)
def run_ml_analysis(self, req_id: str, file_id: str, filename: str, features: dict, username: str):
    """Asynchronous background task to run the machine learning models and update request records."""
    logger.info(f"Background Job: Starting ML analysis for request {req_id} (File ID: {file_id})...")
    
    # 1. Update status to 'processing'
    try:
        logger.info(f"Background Job: Setting request status to 'processing' for {req_id}...")
        db.update_request_status_only(req_id, "processing")
    except Exception as db_err:
        logger.error(f"Background Job: Failed to set request status to 'processing': {db_err}", exc_info=True)
        # We can still proceed to scan even if status update failed, or retry
        self.retry(exc=db_err, countdown=5)
        return

    try:
        # 2. Run Machine Learning Risk Models
        logger.info(f"Background Job: Running risk inference models for {filename}...")
        ml_results = ml_analyzer.analyze_risk(features)
        
        # 3. Update Database Request Status to 'pending' and record ML details
        logger.info(f"Background Job: Updating PostgreSQL database record...")
        verdict = str(
            ml_results.get('verdict')
            or ml_results.get('risk_level')
            or ml_results.get('classification')
            or 'review'
        )
        updated = db.update_request_ml_results(
            request_id=req_id,
            ml_verdict=verdict,
            ml_details=ml_results,
            new_status="pending"
        )
        if not updated:
            raise RuntimeError(f"Could not find request record {req_id} in database to update.")
        
        # 4. Add to Cryptographic Ledger
        logger.info(f"Background Job: Emitting cryptographic ledger transaction event...")
        audit_ledger.add_event(f"User '{username}' uploaded file: {filename} (ID: {file_id})")
        db.log_activity(username, "file_upload", f"File: {filename}")
        
        logger.info(f"Background Job: ML Analysis complete for request {req_id}.")
        return {
            'status': 'SUCCESS',
            'request_id': req_id,
            'ml_analysis': ml_results
        }
    except Exception as exc:
        logger.error(f"Background Job: Task execution error for request {req_id}: {exc}", exc_info=True)
        
        # Check if we have exhausted all retries
        if self.request.retries >= self.max_retries:
            logger.error(f"Background Job: Task failed after max retries. Marking request {req_id} as failed.")
            try:
                db.update_request_status_only(req_id, "failed")
            except Exception as db_err:
                logger.error(f"Background Job: Failed to update request to 'failed' state: {db_err}", exc_info=True)
            raise exc
        
        # Otherwise, attempt retry (transient error / lock / db issues)
        logger.warning(f"Background Job: Retrying task for request {req_id} (Attempt {self.request.retries + 1}/{self.max_retries + 1})...")
        self.retry(exc=exc, countdown=10)
