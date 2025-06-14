from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response: Response = await call_next(request)

        # Add security headers
        response.headers["Strict-Transport-Security"] = "max-age=63072000"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Permissions-Policy"] = "interest-cohort=()"
        response.headers["Content-Security-Policy"] = "frame-ancestors 'none'"

        return response
