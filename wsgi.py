from app import app, start_cleanup_scheduler

# Ensure the background cleanup scheduler is started in the WSGI entry point
start_cleanup_scheduler()

if __name__ == "__main__":
    app.run()
