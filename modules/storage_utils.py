import os
import logging
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class StorageService:
    def __init__(self):
        self.endpoint = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
        self.access_key = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
        self.secret_key = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
        self.bucket_name = os.environ.get("MINIO_BUCKET_NAME", "crypto-uploads")
        self.secure = os.environ.get("MINIO_SECURE", "false").lower() == "true"
        
        protocol = "https" if self.secure else "http"
        self.endpoint_url = f"{protocol}://{self.endpoint}"

        # Initialize boto3 client for S3/MinIO
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name='us-east-1' # Default for MinIO
        )
        self._ensure_bucket()

    def _ensure_bucket(self):
        """Creates the bucket if it does not exist"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                try:
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                    logger.info(f"Bucket '{self.bucket_name}' created.")
                except Exception as ex:
                    logger.error(f"Failed to create bucket: {ex}", exc_info=True)
            else:
                logger.error(f"Error checking bucket: {e}", exc_info=True)
        except Exception as e:
            # MinIO might not be running locally yet
            logger.warning(f"Could not connect to MinIO at {self.endpoint_url}: {e}")

    def upload_file(self, file_id: str, file_data: bytes) -> str:
        """Uploads a file to MinIO and returns the object key (path)"""
        object_key = f"{file_id}.enc"
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=file_data
            )
            return object_key
        except Exception as e:
            # Render production hard-fail requirement
            if os.environ.get("FLASK_ENV") == "production":
                logger.error(f"CRITICAL S3 UPLOAD FAILURE: {e}", exc_info=True)
                raise RuntimeError("Storage upload failed. S3/MinIO is required in production.")
            
            logger.warning(f"Failed to upload to S3: {e}. Falling back to local storage (Development Only).")
            # Fallback to local if MinIO is not running (Only allowed in non-production)
            _BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            UPLOAD_FOLDER = os.path.join(_BASE_DIR, 'uploads')
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            local_path = os.path.join(UPLOAD_FOLDER, object_key)
            with open(local_path, 'wb') as f:
                f.write(file_data)
            return local_path

    def get_file(self, path: str) -> bytes:
        """Retrieves a file from MinIO or local fallback"""
        if path.startswith("uploads") or "/" in path or "\\" in path:
            # It's a local fallback path
            with open(path, 'rb') as f:
                return f.read()
                
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=path)
            return response['Body'].read()
        except Exception as e:
            logger.error(f"Failed to read from S3: {e}", exc_info=True)
            raise Exception("File could not be retrieved from storage.")

    def delete_file(self, path: str) -> bool:
        """Deletes a file from MinIO or local fallback"""
        if not path:
            return False
            
        if path.startswith("uploads") or "/" in path or "\\" in path:
            # It's a local fallback path
            if os.path.exists(path):
                try:
                    os.remove(path)
                    return True
                except OSError as e:
                    logger.error(f"Failed to delete local file: {e}", exc_info=True)
                    return False
            return False

        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=path)
            return True
        except Exception as e:
            logger.error(f"Failed to delete from S3: {e}", exc_info=True)
            return False

storage_service = StorageService()
