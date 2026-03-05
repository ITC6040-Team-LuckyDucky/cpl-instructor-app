#!/bin/bash
set -e

echo "Starting app..."
exec gunicorn --bind=0.0.0.0:${PORT:-8000} --timeout 600 app:app
