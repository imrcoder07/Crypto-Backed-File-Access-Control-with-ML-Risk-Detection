import os
from dotenv import load_dotenv

# Load local env configurations
load_dotenv()

from modules.celery_app import celery_instance
import modules.tasks # Ensure tasks are registered

# Alias for celery command line
celery = celery_instance
