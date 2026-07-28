"""Compatibility entrypoint.

Run `uvicorn app.main:app` for the canonical backend. This module keeps
`uvicorn main:app` working without mounting a second set of legacy routes.
"""

from app.main import app

__all__ = ["app"]
