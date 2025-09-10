🔐 Crypto-Backed File Access Control with ML

(Minor Group Project)

📌 Overview

This project is a secure file access control system that combines Blockchain, Cryptography, and Machine Learning to protect sensitive data. It ensures that file storage and access are encrypted, tamper-proof, and risk-aware.

🔐 Encryption & Decryption: Files are secured using AES/RSA cryptography before storage and sharing.

⛓ Blockchain Integration: Access logs are stored immutably to ensure transparency and tamper-proof records.

🤖 Machine Learning Risk Detection: Anomaly detection models flag suspicious access attempts based on user behavior.

🖥 Web Dashboard: Role-based access control, real-time alerts, and file management via a Flask-powered interface.

✨ Features

Secure file upload and download with cryptographic protection.

Immutable blockchain-based logging of all access requests.

Machine learning model for anomaly and risk detection.

Role-based authentication (Admin, User, Guest).

Flask web dashboard for monitoring, alerts, and file management.

🏗 Project Architecture
User Request → Authentication → File Encryption → Blockchain Logging
                                   ↓
                         ML Risk Detection → Alerts

🛠 Tech Stack

Frontend: HTML, CSS, JavaScript (or React if extended)

Backend: Python (Flask)

Security: AES/RSA Encryption, Blockchain Ledger

Machine Learning: Scikit-learn (Random Forest/Isolation Forest for anomaly detection)

Database: SQLite/MySQL

Version Control: Git & GitHub

⚡ Installation & Setup

Clone the repository

git clone https://github.com/your-username/crypto-backed-file-access-ml.git
cd crypto-backed-file-access-ml


Create a virtual environment & install dependencies

python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
pip install -r requirements.txt


Run the Flask app

python app.py


Open in browser

http://127.0.0.1:5000/

📂 Project Structure
crypto-backed-file-access-ml/
│── app.py                # Main Flask app
│── blockchain.py          # Blockchain logic for access logs
│── encryption.py          # AES/RSA encryption & decryption
│── ml_model.py            # ML anomaly detection
│── static/                # CSS, JS, images
│── templates/             # HTML templates
│── requirements.txt       # Dependencies
│── README.md              # Documentation

📊 Machine Learning Model

Algorithm Used: Random Forest / Isolation Forest

Dataset: Generated synthetic access logs + user behavior patterns

Goal: Flag abnormal file access attempts (e.g., unusual login time, location, or file type).

🚀 Future Enhancements

✅ Integrate real blockchain platforms (Ethereum/Hyperledger).

✅ Improve ML model accuracy with larger datasets.

✅ Add multi-factor authentication (MFA).

✅ Deploy on cloud platforms (AWS, Heroku, or Azure).

👨‍💻 Authors

Your Team Names (Minor Group Project)
