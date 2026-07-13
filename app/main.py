from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import compat_router, router as dubbing_router, stream_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

app = FastAPI(
    title="Auto-Dubbing Video API",
    version="0.1.0",
    description="Synchronous low-VRAM auto-dubbing pipeline for local GPU execution.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:3002",
        "http://127.0.0.1:3002",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dubbing_router)
app.include_router(compat_router)
app.include_router(stream_router)
app.mount("/media", StaticFiles(directory="output"), name="media")


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "auto-dubbing-api", "status": "running"}
