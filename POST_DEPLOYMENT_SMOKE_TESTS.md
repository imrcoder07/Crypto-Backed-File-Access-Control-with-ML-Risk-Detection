# Post-Deployment Smoke Tests

Execute these validation checks immediately after Render successfully deploys the codebase to production.

## 1. Storage Validation (Crucial)
1. Log into the live web application on your custom domain.
2. Upload a test file (e.g., `test_upload.pdf`).
3. **If S3 is misconfigured:** The application will fail loudly with an error page (500), and the Render logs will show `[ERROR] CRITICAL S3 UPLOAD FAILURE`. This means the safety net worked.
4. **If S3 is configured correctly:** The upload will succeed.
5. Verify the file exists inside the external S3/MinIO bucket via your cloud provider's console.

## 2. HTTPS Validation
1. Access the application via `http://` (insecure) instead of `https://`.
2. Ensure the redirect behavior forces the connection back to `https://`.
3. Open the browser's Developer Tools -> Application -> Cookies.
4. Ensure the `session` cookie has the `Secure` flag checked. This proves that `ProxyFix` is successfully intercepting the load balancer protocol headers.

## 3. Background Task Validation
1. Check the Render Log Stream.
2. Confirm you do **not** see `🧹 Cleanup: Removed...` polling every hour on your web instances. 
3. If no cleanup logs appear, the background thread was successfully disabled for the web workers.
