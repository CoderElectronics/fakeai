#!/bin/bash
# Classifier startup script for production service
# by: Ari Stehney

git pull

# run latest script
source ./.venv/bin/activate
gunicorn -w 4 -b 0.0.0.0:8003 classifier_server:app

echo Server exited!