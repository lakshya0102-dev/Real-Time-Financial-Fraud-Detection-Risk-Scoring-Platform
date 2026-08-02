"""Uvicorn entry point for the Fraud Detection API."""

import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.app import app

if __name__ == "__main__":
    import uvicorn
    from src.config.settings import get_settings

    settings = get_settings()
    uvicorn.run(
        "src.api.app:app",
        host=settings.api.host,
        port=settings.api.port,
        workers=settings.api.workers,
        reload=settings.api.debug,
    )
