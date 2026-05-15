from datetime import datetime, timezone
import re

def get_time_ago(timestamp):
    if not timestamp:
        return "Never"
    if isinstance(timestamp, str):
        # Fix CQ-8: handle strings safely with timezone-awareness if they end with 'Z'
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            # Fallback for unexpected formats
            return str(timestamp)
    else:
        dt = timestamp
        
    # Make 'now' aware if the parsed dt is aware
    if dt.tzinfo is not None:
        now = datetime.now(timezone.utc)
    else:
        now = datetime.now()
        
    diff = now - dt
    
    if diff.days > 0:
        return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
    elif diff.seconds >= 3600:
        hours = diff.seconds // 3600
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    elif diff.seconds >= 60:
        minutes = diff.seconds // 60
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    else:
        return "Just now"

def validate_filename(filename):
    if not filename or not isinstance(filename, str):
        return False
    
    if '..' in filename or '/' in filename or '\\' in filename:
        return False
    
    if any(ord(c) < 32 for c in filename):
        return False
    
    # Only block dangerous executables, allow all documents
    dangerous_extensions = ['.exe', '.bat', '.cmd', '.ps1', '.scr', '.dll', '.sh', '.bin', '.jar']
    if any(filename.lower().endswith(ext) for ext in dangerous_extensions):
        return False
    
    return True
