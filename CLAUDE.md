# Crypto Access Control App Instructions

## 1. Project Overview (WHY)
This app is a secure file access control system for admins and business users who need protected file uploads, approval workflows, audit logging, and ML-assisted risk checks. It combines encrypted file handling, role-based access, and an internal blockchain-style activity log in a single Flask application.

## 2. Tech Stack (WHAT)
- Framework: Flask
- Language: Python 3.10, JavaScript, HTML, CSS
- Database: None; in-memory Python storage and local filesystem uploads
- Styling: Tailwind CSS, custom CSS, Chart.js, Three.js
- DO NOT USE: Do not introduce Redux, React, or a separate ORM/database layer unless explicitly requested

## 3. Common Commands (HOW)
- Install: `pip install -r requirements.txt`
- Dev: `python app.py`
- Production: `gunicorn app:app`
- Docker Build: `docker build -t crypto-access-control-app .`
- Test ML Models: `python Crypto-models/test_models.py`

## 4. Architecture & Structure
- `/app.py`: Main Flask application, route handlers, in-memory storage, ML analyzer, encryption flow, and session logic.
- `/templates/index.html`: Single-page frontend containing the landing page, user dashboard, admin dashboard, and client-side JavaScript.
- `/modules`: Small helper modules for encryption and blockchain utilities.
- `/Crypto-models`: Trained model artifacts, test script, notebook, and ML visualizations.
- `/uploads`: Encrypted uploaded files stored on disk.

Put backend/API changes in `app.py` unless a new helper module clearly reduces complexity. Put UI changes in `templates/index.html` unless the frontend is being intentionally split into smaller assets. Keep ML artifacts and experiments inside `Crypto-models` to avoid mixing them into runtime code.

## 5. Coding Conventions
- Use simple, explicit Python functions and keep route logic readable.
- Prefer small helper functions when repeated logic appears in multiple routes.
- Preserve the current Flask + single-template structure unless a refactor is explicitly requested.
- All file access, encryption, and download flows must include error handling and user-safe error responses.
- Keep security-sensitive logic straightforward and avoid hidden side effects.
- Validate filenames and request inputs before processing files.

## 6. Rules & Prohibitions
- NEVER delete or overwrite files in `/uploads` unless the task explicitly requires it.
- ALWAYS run at least a syntax check or relevant verification before claiming backend changes are done.
- Ask for confirmation before installing new Python or npm packages.
- Do not add a database, external auth provider, or major framework migration unless explicitly requested.
