from fastapi import Request, Depends, status, HTTPException
from fastapi.responses import RedirectResponse
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# In-memory session store (replace with database in production)
sessions: dict[str, str] = {}  # session_id: email

# Dependency to check authentication
async def get_current_user(request: Request) -> Optional[str]:
    session_id = request.cookies.get("session_id")
    if session_id and session_id in sessions:
        logger.debug(f"Found session_id: {session_id}, user: {sessions[session_id]}")
        return sessions[session_id]
    logger.debug("No valid session_id found")
    return None

# Middleware to protect routes
async def require_auth(request: Request, current_user: Optional[str] = Depends(get_current_user)):
    public_paths = ["/", "/login", "/register", "/auth/register", "/auth/verify-register", "/auth/resend-otp", "/auth/login"]
    is_api_request = request.url.path.startswith("/get_unique_id") or "application/json" in request.headers.get("accept", "").lower()
    
    if not current_user and request.url.path not in public_paths:
        if is_api_request:
            logger.debug(f"API request to {request.url.path} unauthorized, raising 401")
            raise HTTPException(status_code=401, detail="Authentication required")
        logger.debug(f"Redirecting to /login from {request.url.path}")
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    return current_user