import os
from celery import Celery

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")

# Format secure Redis SSL requirements for Celery if using rediss://
CELERY_REDIS_URL = REDIS_URL
if CELERY_REDIS_URL.startswith("rediss://") and "ssl_cert_reqs" not in CELERY_REDIS_URL:
    separator = "&" if "?" in CELERY_REDIS_URL else "?"
    CELERY_REDIS_URL = f"{CELERY_REDIS_URL}{separator}ssl_cert_reqs=CERT_NONE"

def make_celery(app_name="crypto_access_control"):
    celery_app = Celery(
        app_name,
        broker=CELERY_REDIS_URL,
        backend=CELERY_REDIS_URL
    )
    
    # Namespace separation & production-safe configurations
    celery_app.conf.update(
        broker_transport_options={'global_keyprefix': 'celery_broker:'},
        result_key_prefix='celery_result:',
        
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        
        # Optimize prefetching for small plans (prevents worker from hoarding tasks)
        worker_prefetch_multiplier=1,
    )
    return celery_app

celery_instance = make_celery()
