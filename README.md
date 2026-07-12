# 🔐 Crypto Access Control

<div align="center">

**A production-grade, ML-enhanced cryptographic file access control platform.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-crypto--access.in-6366f1?style=for-the-badge&logo=render)](https://crypto-access.in)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1.2-000000?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgresql.org)
[![Redis](https://img.shields.io/badge/Redis-Upstash-DC382D?style=for-the-badge&logo=redis&logoColor=white)](https://upstash.com)
[![Celery](https://img.shields.io/badge/Celery-5.3.6-37814A?style=for-the-badge&logo=celery)](https://docs.celeryq.dev/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikitlearn)](https://scikit-learn.org/)

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Technology Stack](#-technology-stack)
- [ML Risk Detection](#-ml-risk-detection)
- [Async Processing Pipeline](#-async-processing-pipeline)
- [Security Implementation](#-security-implementation)
- [API Reference](#-api-reference)
- [Database Schema](#-database-schema)
- [Deployment](#-deployment)
- [Local Development Setup](#-local-development-setup)
- [Environment Variables](#-environment-variables)
- [Running Tests](#-running-tests)
- [Future Enhancements](#-future-enhancements)

---

## 🌟 Project Overview

**Crypto Access Control** is a zero-trust file management system that combines **AES-256 server-side encryption**, **ML ensemble risk detection**, and a **cryptographic audit ledger** into a unified platform.

The system enforces a strict approval workflow: no file is accessible until an administrator explicitly grants permission, guided by machine learning risk scores from three independent models.

### Motivation

Traditional file storage systems offer insufficient protection for sensitive data at rest. This project addresses:

- **Data-at-rest vulnerability** → Mitigated via AES-256 encryption of every uploaded file
- **Insider threat risk** → Mitigated via RBAC, audit logging, and admin-gated access
- **Unknown file threats** → Mitigated via real-time ML risk scoring before admin review
- **Audit gaps** → Mitigated via a tamper-evident cryptographic ledger

---

## ❓ Problem Statement

Organizations managing confidential documents face three critical challenges:

1. **Unauthorized access** – Files accessible to unauthorized personnel
2. **Unknown threat vectors** – Malicious files without detection mechanisms
3. **No auditability** – No reliable trail of who accessed what, and when

This system solves all three simultaneously through an integrated pipeline.

---

## ✨ Key Features

### 🔐 Security & Authentication
- Role-Based Access Control (RBAC) with Admin/User separation
- Bcrypt-hashed passwords with secure session management
- AES-256 server-side file encryption via the `cryptography` library
- Secure HTTPS-only sessions with `SameSite=Lax` CSRF protection
- ProxyFix middleware for secure Render/reverse-proxy operation
- SHA-256 file fingerprinting for off-site file tampering/modification detection

### 🤖 AI-Powered Risk Detection
- **Ensemble ML** using Random Forest + SVM + Isolation Forest
- SHA-256 model integrity verification before loading
- Asynchronous inference via Celery background workers
- Automatic fallback to safe mock scores when models unavailable
- Risk classification: Low / Medium / High → Admin review guidance

### ⛓️ Cryptographic Audit Ledger
- Immutable, chained event log for all system actions
- SHA-256 hash-linked entries (blockchain-inspired)
- Covers: uploads, approvals, denials, downloads, admin actions
- Tamper-evident design — hash mismatch detection

### 🚀 Async Processing Architecture
- HTTP 202 immediate acknowledgement on upload
- Redis-backed Celery task queue (Upstash Redis)
- Request lifecycle: `QUEUED → PROCESSING → PENDING → APPROVED/DENIED`
- Frontend status polling every 2 seconds
- Automatic retry with exponential backoff (max 3 retries)
- `FAILED` status after retry exhaustion

### 📊 Admin Dashboard
- Real-time system statistics and analytics
- Pending request management with ML risk scores
- Approved file management and access control
- User management interface
- Blockchain audit log viewer

### 👤 User Dashboard
- File upload with real-time status tracking
- My Requests view with approval status
- Activity history
- Download approved files securely

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Render Web Service                           │
│                                                                 │
│  ┌──────────────┐    ┌─────────────────────────────────────┐   │
│  │   Gunicorn   │    │         Flask Application           │   │
│  │  (WSGI HTTP) │───▶│  ┌──────────┐  ┌────────────────┐  │   │
│  │              │    │  │  Routes  │  │   Blueprints   │  │   │
│  └──────────────┘    │  │ /auth    │  │ auth_bp        │  │   │
│                       │  │ /user    │  │ user_bp        │  │   │
│  ┌──────────────┐    │  │ /admin   │  │ admin_bp       │  │   │
│  │ Celery Worker│    │  │ /upload  │  │ main_bp        │  │   │
│  │  (solo pool) │    │  └──────────┘  └────────────────┘  │   │
│  │              │    └─────────────────────────────────────┘   │
│  └──────┬───────┘                                              │
└─────────┼────────────────────────────────────────────────────-─┘
          │
          ▼ Async Task Queue
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  Upstash Redis  │   │   Render        │   │   AWS S3 /      │
│  (Task Broker)  │   │   PostgreSQL    │   │   MinIO Storage │
│  • Celery queue │   │   (Primary DB)  │   │   (Encrypted    │
│  • Rate limiter │   │   • Users       │   │    Files)       │
│    keys (prefix │   │   • Requests    │   │                 │
│    separated)   │   │   • Activities  │   │                 │
└─────────────────┘   │   • Audit Log   │   └─────────────────┘
                       └─────────────────┘
```

### Upload Flow (Async Mode)

```
User Browser
    │
    │  POST /upload (multipart)
    ▼
Flask Route (user.py)
    │
    ├─ 1. Validate file & user session
    ├─ 2. Encrypt file (AES-256)
    ├─ 3. Upload encrypted file to S3
    ├─ 4. Create Request Stub (status=QUEUED) ← Synchronous
    ├─ 5. Enqueue Celery Task → Redis
    └─ 6. Return HTTP 202 { task_id, request_id }
              │
              ▼ (background)
    Celery Worker (tasks.py)
    │
    ├─ 7.  Update DB: status = PROCESSING
    ├─ 8.  Load ML models (RF + SVM + IsolationForest)
    ├─ 9.  Run generate_features() → feature vector
    ├─ 10. Run analyze_risk() → ensemble score
    ├─ 11. Update DB: status = PENDING, ml_verdict, ml_details
    └─ 12. Write to Cryptographic Audit Ledger
              │
              ▼ (browser polling every 2s)
    GET /api/request_status/<request_id>
    │
    └─ Return { status, ml_verdict, ml_details }

Admin Dashboard
    │
    └─ Review pending requests with ML scores
       → Approve / Deny → APPROVED / DENIED
```

---

## 🛠️ Technology Stack

### Backend
| Component | Technology | Version |
|-----------|-----------|---------|
| Web Framework | Flask | 3.1.2 |
| WSGI Server | Gunicorn | 21.2.0 |
| Database Driver | psycopg2-binary | 2.9.9 |
| Task Queue | Celery | 5.3.6 |
| Message Broker | Redis (Upstash) | 8.0.0 |
| DB Migrations | Flask-Migrate / Alembic | 4.0.5 |
| Rate Limiting | Flask-Limiter | 3.7.0 |
| File Storage | boto3 (S3/MinIO) | 1.34.84 |
| Encryption | cryptography (AES-256) | 42.0.5 |
| Password Hashing | bcrypt | 4.1.2 |
| Env Management | python-dotenv | 1.0.1 |

### Machine Learning
| Model | Library | Purpose |
|-------|---------|---------|
| Random Forest | scikit-learn 1.4.2 | Probability-based classification |
| SVM (RBF kernel) | scikit-learn 1.4.2 | Boundary classification |
| Isolation Forest | scikit-learn 1.4.2 | Anomaly/outlier detection |
| Feature Engineering | numpy 1.26.4 / pandas 2.3.2 | Data preprocessing |
| Model Serialization | joblib 1.4.2 | Pickle pipelines with integrity hashing |

### Frontend
| Technology | Purpose |
|-----------|---------|
| HTML5 / CSS3 | Structure & Styling |
| Tailwind CSS | Utility-first responsive design |
| Three.js | 3D animated background (plexus network) |
| Chart.js | Analytics dashboards |
| Vanilla JavaScript (ES6+) | API integration & polling |

### Infrastructure
| Service | Provider | Role |
|---------|---------|------|
| Web App | Render | Flask + Gunicorn + Celery (hybrid) |
| Database | Render PostgreSQL | Primary data store |
| Cache & Queue | Upstash Redis | Celery broker + result backend |
| File Storage | S3-Compatible (MinIO/AWS) | Encrypted file storage |

---

## 🤖 ML Risk Detection

### Feature Vector

Each file upload generates the following 6-dimensional feature vector:

```python
{
    'file_size': int,           # Raw bytes
    'name_length': int,         # Length of filename
    'is_executable': 0 | 1,    # .exe .dll .bat .sh .bin
    'has_special_chars': 0 | 1, # Non-alphanumeric in name
    'upload_hour': 0–23,        # Hour of upload (behaviour)
    'user_trust_score': float   # Historical user trust (0–1)
}
```

### Ensemble Decision Logic

```
RF Risk Probability  →  P_rf   ∈ [0, 1]
SVM Risk Probability →  P_svm  ∈ [0, 1]
Isolation Forest     →  iso_pred ∈ {1 (inlier), -1 (outlier)}

Ensemble Score = (P_rf + P_svm) / 2
If iso_pred == -1 (anomaly):
    Ensemble Score = min(1.0, Ensemble Score + 0.30)

Risk Threshold: is_risky = (Ensemble Score > 0.75)
```

### Model Integrity

All models are protected via **SHA-256 hash verification** at load time:

```bash
python Crypto-models/generate_model_hashes.py   # Run once after training
```

A hash mismatch prevents model loading and triggers a critical security alert.

---

## ⚡ Async Processing Pipeline

The system supports two operating modes:

| Mode | Config | Behavior |
|------|--------|---------|
| **Async** (Production) | `USE_ASYNC_ML=true` | HTTP 202, Celery worker, polling |
| **Sync** (Fallback) | `USE_ASYNC_ML=false` | Blocking inline ML inference |

### Request Lifecycle States

```
QUEUED      → Request created, task enqueued
PROCESSING  → Celery worker executing ML inference
PENDING     → ML complete, awaiting admin decision
APPROVED    → Admin approved, file accessible
DENIED      → Admin denied
FAILED      → Worker exhausted all retries (max 3)
```

### Data Integrity Guarantee

The request stub is created **synchronously before** the Celery task is enqueued. This guarantees:
- S3 file will always have a matching database record
- No orphaned files on queue failure
- Worker outages produce `QUEUED` records, not silent data loss

---

## 🔒 Security Implementation

### Layered Security Model

```
Layer 1 – Transport        HTTPS enforced (SESSION_COOKIE_SECURE=True)
Layer 2 – Authentication   Bcrypt password hashing, session management
Layer 3 – Authorization    RBAC (admin vs. user), route-level guards
Layer 4 – Encryption       AES-256 server-side file encryption
Layer 5 – Validation       Flask-Limiter rate limits on auth endpoints
Layer 6 – Integrity        SHA-256 model hash verification
Layer 7 – Audit            Cryptographic hash-chained event ledger
```

### Session Security

```python
SESSION_COOKIE_SECURE   = True   # HTTPS only
SESSION_COOKIE_HTTPONLY = True   # No JS access
SESSION_COOKIE_SAMESITE = 'Lax'  # CSRF protection
```

### Redis Namespace Isolation

```
celery-broker:  redis://...?  → Celery task queue keys
celery-backend: redis://...?  → Celery result backend keys
FLASK_LIMITER:  redis://...?  → Rate limiting keys (key_prefix)
```

---

## 📡 API Reference

### Authentication

| Method | Endpoint | Description |
|--------|---------|-------------|
| `POST` | `/api/login` | User login |
| `POST` | `/api/signup` | User registration |
| `POST` | `/api/logout` | Session logout |

### User Endpoints

| Method | Endpoint | Description |
|--------|---------|-------------|
| `POST` | `/upload` | Upload encrypted file (async) → HTTP 202 |
| `GET` | `/api/request_status/<id>` | Poll async task status |
| `GET` | `/api/user/files` | List user's files and requests |
| `GET` | `/api/user/activities` | User activity log |
| `GET` | `/download/<file_id>` | Download approved file |
| `GET` | `/api/profile` | Get user profile |
| `POST` | `/api/profile` | Update profile |

### Admin Endpoints

| Method | Endpoint | Description |
|--------|---------|-------------|
| `GET` | `/api/system_stats` | System-wide statistics |
| `GET` | `/api/admin/pending_requests` | All pending requests with ML |
| `POST` | `/api/admin/approve_request` | Approve a file request |
| `POST` | `/api/admin/deny_request` | Deny a file request |
| `GET` | `/api/admin/approved_files` | All approved files |
| `GET` | `/api/admin/users` | User management |
| `GET` | `/api/blockchain_log` | Full cryptographic audit log |

---

## 🗄️ Database Schema

### `users` table
```sql
id          SERIAL PRIMARY KEY
username    VARCHAR(80) UNIQUE NOT NULL
password    TEXT NOT NULL          -- bcrypt hash
role        VARCHAR(20) DEFAULT 'user'
trust_score FLOAT DEFAULT 0.85
created_at  TIMESTAMP
```

### `requests` table
```sql
id           UUID PRIMARY KEY
user_id      INTEGER REFERENCES users(id)
filename     VARCHAR(255)
file_id      VARCHAR(255)          -- S3 object key
file_size    BIGINT                -- bytes
file_size_mb FLOAT                 -- display value
status       VARCHAR(50)           -- QUEUED/PROCESSING/PENDING/APPROVED/DENIED/FAILED
ml_verdict   TEXT
ml_details   JSONB                 -- full ensemble results
created_at   TIMESTAMP
updated_at   TIMESTAMP
```

### `activities` table
```sql
id          SERIAL PRIMARY KEY
username    VARCHAR(80)
action      VARCHAR(100)
detail      TEXT
timestamp   TIMESTAMP
```

### `audit_log` table
```sql
id          SERIAL PRIMARY KEY
event       TEXT
hash        VARCHAR(64)           -- SHA-256 chain hash
previous_hash VARCHAR(64)
timestamp   TIMESTAMP
```

---

## 🚀 Deployment

### Production Stack (Render)

```
render.yaml
├── web: Flask + Gunicorn + Celery Worker (hybrid, single container)
│   └── entrypoint.sh
│       ├── alembic upgrade head          # Migrations
│       ├── python bootstrap_admin.py     # Admin seed
│       ├── celery -A worker worker \     # Background (if USE_ASYNC_ML=true)
│       │         --pool=solo &
│       └── exec gunicorn app:app \       # Foreground
│                 --bind 0.0.0.0:$PORT
└── PostgreSQL (managed Render DB)
```

### Render Environment Variables

```bash
DATABASE_URL=postgresql://...
SECRET_KEY=<strong-random-key>
S3_ENDPOINT_URL=https://...
S3_ACCESS_KEY=...
S3_SECRET_KEY=...
S3_BUCKET_NAME=...
REDIS_URL=rediss://...        # Upstash TLS URL
USE_ASYNC_ML=true
ADMIN_USERNAME=admin
ADMIN_PASSWORD=<secure-password>
FLASK_ENV=production
```

---

## 💻 Local Development Setup

### Prerequisites

- Python 3.11+
- PostgreSQL (local or Docker)
- Redis (local or Upstash account)

### 1. Clone the Repository

```bash
git clone https://github.com/<your-org>/crypto-access-control.git
cd crypto-access-control
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your local credentials
```

### 5. Run Migrations

```bash
alembic upgrade head
```

### 6. Bootstrap Admin Account

```bash
python manage_admin.py
```

### 7. Start the Application

```bash
# Sync mode (no Redis required)
USE_ASYNC_ML=false python app.py

# Async mode (requires Redis + Celery)
celery -A worker worker --pool=solo --loglevel=info &
python app.py
```

### 8. Access the Application

```
http://localhost:5000
```

---

## 🔧 Environment Variables

| Variable | Required | Description |
|---------|---------|-------------|
| `DATABASE_URL` | ✅ | PostgreSQL connection string |
| `SECRET_KEY` | ✅ | Flask session signing key |
| `S3_ENDPOINT_URL` | ✅ | S3-compatible storage endpoint |
| `S3_ACCESS_KEY` | ✅ | S3 access key |
| `S3_SECRET_KEY` | ✅ | S3 secret key |
| `S3_BUCKET_NAME` | ✅ | Storage bucket name |
| `REDIS_URL` | ✅ | Redis/Upstash connection URL |
| `USE_ASYNC_ML` | ✅ | `true` / `false` — Async mode toggle |
| `ADMIN_USERNAME` | Optional | Auto-bootstrap admin on startup |
| `ADMIN_PASSWORD` | Optional | Auto-bootstrap admin password |
| `FLASK_ENV` | Optional | `production` or `development` |
| `PORT` | Optional | Port for Gunicorn (default: 8000) |
| `RUN_BACKGROUND_TASKS` | Optional | `true` to enable background cleanup |

---

## 🧪 Running Tests

```bash
# Run full test suite
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --tb=short

# Run specific test file
pytest tests/test_integration.py -v
```

### Test Coverage

The integration test suite covers:
- ✅ User registration and login
- ✅ File upload (sync and async modes)
- ✅ Async request status polling
- ✅ Admin approval workflow
- ✅ File download with encryption/decryption
- ✅ Blockchain audit log integrity
- ✅ Rate limiting validation
- ✅ RBAC enforcement
- ✅ Prevents duplicate uploads (SHA-256 fingerprint check)
- ✅ Identifies/flags off-site file tampering on re-upload

---

## 📁 Project Structure

```
crypto-access-control/
├── app.py                     # Flask application factory
├── wsgi.py                    # WSGI entry point
├── worker.py                  # Celery worker entry point
├── entrypoint.sh              # Docker/Render startup script
├── requirements.txt           # Python dependencies
├── alembic.ini                # Alembic migration config
├── render.yaml                # Render deployment config
│
├── modules/
│   ├── db.py                  # PostgreSQL persistence layer
│   ├── ml_analyzer.py         # ML ensemble risk engine
│   ├── tasks.py               # Celery async task definitions
│   ├── celery_app.py          # Celery instance & configuration
│   ├── auth_utils.py          # Authentication utilities
│   ├── audit_utils.py         # Cryptographic audit ledger
│   ├── encryption_utils.py    # AES-256 file encryption
│   ├── storage_utils.py       # S3/MinIO storage abstraction
│   ├── extensions.py          # Shared Flask extensions
│   └── utils.py               # General utilities
│
├── routes/
│   ├── auth.py                # /login, /signup, /logout
│   ├── main.py                # Landing page
│   ├── user.py                # /upload, /download, user APIs
│   └── admin.py               # Admin management APIs
│
├── templates/
│   └── index.html             # Single-page application shell
│
├── migrations/                # Alembic migration scripts
├── Crypto-models/             # Trained ML model artifacts
│   ├── models/
│   │   ├── random_forest_pipeline.pkl
│   │   ├── svm_pipeline.pkl
│   │   └── isolation_forest_pipeline.pkl
│   └── model_hashes.json      # SHA-256 integrity hashes
│
└── tests/
    └── test_integration.py    # Full integration test suite
```

## 🆕 Recent Updates & Quality Assurance

The system underwent a comprehensive security hardening, frontend restoration, and quality assurance review. The following features were added, verified, and integrated:

### 1. Frontend File Upload & Status Polling
- **Integrated Submit Listener:** Restored the `#upload-form` submit handler in the user dashboard. Users can now select files, set passwords, and trigger uploads without experiencing interface hangs.
- **Dynamic Celery Status Polling:** Fully connected the frontend to `/api/request_status/<request_id>` to poll risk assessment progress every 2 seconds with clear status alerts (`QUEUED → PROCESSING → SUCCESS/FAILURE`).
- **Complete Test Coverage:** Verified integration and async upload pathways using the pytest suite, passing all 10 integration and unit tests.

### 2. B.Tech Thesis Formatting & Quality Assurance Review
- **Standardized Margins for Printing:** Configured all thesis sections with standard margins (Left = 1.25" for binding clearance, Right/Top/Bottom = 1.0") in `CRYPTO thesis final year_Final_Reviewed.docx`.
- **Spelling & Terminology Audit:** Audited and corrected 17 occurrences of the spelling typo `Isolation Forestst` to `Isolation Forest`, corrected `ASANSOL ENGINEERING COLLAGE` to `ASANSOL ENGINEERING COLLEGE`, and aligned figure captions with their respective list indices.
- **Dynamic Field Refresh:** Embedded the `w:updateFields` Word settings property to prompt automatic index and page-number refreshes on startup.

---

## 🔮 Future Enhancements

| Feature | Priority | Description |
|---------|---------|-------------|
| Two-Factor Authentication (2FA) | High | TOTP-based 2FA for admin accounts |
| File Preview | Medium | Secure in-browser preview for approved files |
| Dedicated Celery Worker | Medium | Separate Koyeb/Railway worker service |
| Advanced ML Features | Low | File content hash, entropy analysis, MIME validation |
| WebSocket Status Updates | Low | Replace polling with WebSocket push |
| Bulk Operations | Low | Admin bulk approve/deny |
| Mobile Application | Future | React Native mobile client |

---

## 📄 License

This project is designed for academic research and enterprise security applications. All rights reserved.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'feat: add your feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Open a Pull Request

**Security Issues**: Report security vulnerabilities privately via the repository's security advisory channel. Do not open public issues for security bugs.

---

<div align="center">

Built with ❤️ using Flask, PostgreSQL, Celery, Redis, and scikit-learn.

**Live at [crypto-access.in](https://crypto-access.in)**

</div>
