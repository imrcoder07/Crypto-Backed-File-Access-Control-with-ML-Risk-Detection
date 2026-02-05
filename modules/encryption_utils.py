from cryptography.fernet import Fernet

def generate_key():
    """
    Generates a new, unique encryption key for a single file.
    This is called every time a new file is uploaded.
    """
    return Fernet.generate_key()

def encrypt_file_data(data, key):
    """
    Encrypts the raw byte data of a file using its unique key.
    Returns the encrypted byte data.
    """
    fernet = Fernet(key)
    return fernet.encrypt(data)

def decrypt_file_data(encrypted_data, key):
    """
    Decrypts the raw byte data of a file using its unique key.
    Returns the original, decrypted byte data.
    """
    fernet = Fernet(key)
    try:
        # This will fail if the key is incorrect, preventing unauthorized access.
        return fernet.decrypt(encrypted_data)
    except Exception as e:
        # Log the error; in a real app, this might trigger a security alert.
        print(f"Decryption failed: {e}")
        return None

