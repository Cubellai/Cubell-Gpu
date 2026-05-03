#!/usr/bin/env sh
set -eu

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."

exec celery -A cubell.gpu_worker worker \
  --loglevel="${LOG_LEVEL:-INFO}" \
  --queues="${DUBBING_QUEUE:-dubbing}" \
  --concurrency="${WORKER_CONCURRENCY:-1}" \
  --prefetch-multiplier=1
