import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import abc

class BaseEncryptionService(abc.ABC):
    """Abstract interface for encryption services (supports future AES-GCM migration)"""
    @abc.abstractmethod
    def encrypt(self, data: bytes, password: str) -> tuple[bytes, bytes]:
        pass
        
    @abc.abstractmethod
    def decrypt(self, encrypted_data: bytes, password: str, salt: bytes) -> bytes:
        pass

class AdvancedEncryption(BaseEncryptionService):
    @staticmethod
    def generate_key_from_password(password: str, salt: bytes = None) -> tuple:
        """Generate encryption key from user password"""
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key, salt
    
    def encrypt(self, data: bytes, password: str) -> tuple[bytes, bytes]:
        """Encrypt data with user password"""
        key, salt = self.generate_key_from_password(password)
        fernet = Fernet(key)
        encrypted_data = fernet.encrypt(data)
        return encrypted_data, salt
    
    def decrypt(self, encrypted_data: bytes, password: str, salt: bytes) -> bytes:
        """Decrypt data with user password"""
        try:
            key, _ = self.generate_key_from_password(password, salt)
            fernet = Fernet(key)
            return fernet.decrypt(encrypted_data)
        except Exception as e:
            # We raise a generic exception to hide raw crypto stack traces (UX-5)
            raise Exception("Decryption failed. The password may be incorrect or the file may be corrupted.")

# The default active service
encryption_service = AdvancedEncryption()
