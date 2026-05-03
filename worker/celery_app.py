"""Compatibility wrapper for older worker.celery_app imports.

Use ``cubell.gpu_worker`` as the Celery app entrypoint for new worker startup
commands.
"""

from cubell.gpu_worker import app, celery, celery_app

__all__ = ["app", "celery", "celery_app"]
