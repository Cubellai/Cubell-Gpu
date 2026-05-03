# LipDub AI - Dubbing Platform

## How to start

1. Copy env file
   cp .env.example .env

2. Install dependencies
   pip install -r requirements.txt

3. Start Redis (in another terminal)
   redis-server

4. Start the worker
   ./start_worker.sh

5. Start the FastAPI backend
   uvicorn app.main:app --reload
