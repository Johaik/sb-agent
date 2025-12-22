from celery import Celery
from src.config import Config

celery_app = Celery(
    "research_worker",
    broker=Config.REDIS_URL,
    backend=Config.REDIS_URL,
    include=["src.worker.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)
