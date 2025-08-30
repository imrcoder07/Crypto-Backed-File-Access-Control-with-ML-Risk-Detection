# encryption.py

from cryptography.fernet import Fernet

def generate_key(key_file='filekey.key'):
    """
    Generates a new key and saves it to a file.
    Run this once and securely store the key!
    """
    key = Fernet.generate_key()
    with open(key_file, 'wb') as f:
        f.write(key)
    print(f"Encryption key generated and saved to {key_file}.")

def load_key(key_file='filekey.key'):
    """
    Loads the secret key from a file.
    """
    with open(key_file, 'rb') as f:
        key = f.read()
    return key

def encrypt_file(filename, key_file='filekey.key'):
    """
    Encrypts a file with the loaded key.
    """
    key = load_key(key_file)
    fernet = Fernet(key)
    with open(filename, 'rb') as file:
        original = file.read()
    encrypted = fernet.encrypt(original)
    with open(filename, 'wb') as encrypted_file:
        encrypted_file.write(encrypted)
    print(f"{filename} encrypted successfully.")

def decrypt_file(filename, key_file='filekey.key'):
    """
    Decrypts a file with the loaded key.
    """
    key = load_key(key_file)
    fernet = Fernet(key)
    with open(filename, 'rb') as file:
        encrypted = file.read()
    decrypted = fernet.decrypt(encrypted)
    with open(filename, 'wb') as decrypted_file:
        decrypted_file.write(decrypted)
    print(f"{filename} decrypted successfully.")

# Example Usage:
if __name__ == "__main__":
    # Only run to generate a key ONCE
    # generate_key()

    # To encrypt or decrypt files, first ensure the key exists.
    # encrypt_file("example.txt")
    # decrypt_file("example.txt")
    pass
