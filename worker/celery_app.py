from celery import Celery

from worker.config import get_settings

settings = get_settings()

celery_app = Celery(
    "cubell_gpu_worker",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["worker.tasks"],
)

celery_app.conf.update(
    task_acks_late=True,
    task_default_queue=settings.dubbing_queue,
    task_reject_on_worker_lost=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,
    timezone="UTC",
)
