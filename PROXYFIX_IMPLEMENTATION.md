# ProxyFix Implementation for Reverse Proxy Awareness

**Purpose:** 
To ensure the Flask application accurately interprets incoming traffic headers when running behind Render's reverse proxy/load balancer.

**Production Risk Addressed:**
Render terminates HTTPS traffic at its load balancer. This means that internally, requests forwarded to Gunicorn appear as standard HTTP traffic. Without middleware, Flask will incorrectly assume the connection is insecure. This breaks secure session configurations (like `SESSION_COOKIE_SECURE = True`), causes HTTPS redirects to loop or fail, and causes URL generators (`url_for`) to generate insecure `http://` links.

**Implementation Details:**
- Imported `from werkzeug.middleware.proxy_fix import ProxyFix` in `app.py`.
- Wrapped the core WSGI application: `app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)`.
- **x_proto=1:** Instructs Flask to trust the `X-Forwarded-Proto` header from exactly one proxy hop (Render), ensuring HTTPS detection works.
- **x_host=1:** Instructs Flask to trust the `X-Forwarded-Host` header from exactly one hop, ensuring URL generation uses the correct custom domain rather than internal Docker routing IPs.

**Operational Verification Steps:**
1. After deployment, navigate to the live custom domain.
2. Log into the application and inspect the session cookie in your browser's Developer Tools.
3. Verify that the `Secure` flag is `True`. (If ProxyFix fails, Flask will refuse to set this cookie over what it believes is an HTTP connection, and logins will break).
