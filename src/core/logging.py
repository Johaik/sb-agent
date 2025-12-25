import structlog
import logging
import sys

def setup_logging():
    # Configure standard logging to use structlog
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=logging.INFO)
    
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )

