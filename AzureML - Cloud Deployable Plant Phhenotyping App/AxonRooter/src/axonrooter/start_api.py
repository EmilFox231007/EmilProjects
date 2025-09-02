"""
Entrypoint script to launch the AxonRooter FastAPI application using Uvicorn.

Configures logging with detailed debug level output to both console and file,
then starts the ASGI server on all interfaces (0.0.0.0) at port 80.

Imports the FastAPI app instance and logging setup utilities from the axonrooter package.

Usage:
    python -m axonrooter.main

Note:
    The Uvicorn internal logging configuration is disabled (log_config=None)
    to use the custom logging setup instead.
"""

import logging

import uvicorn

from axonrooter.app.app import app
from axonrooter.logging_config import configure_logging, setup_logging

configure_logging(enabled=True, level=logging.DEBUG)
setup_logging()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80, log_config=None)
