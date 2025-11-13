#!/bin/sh
PORT=${PORT:-10000}
exec uvicorn main:app --host 0.0.0.0 --port "$PORT"

