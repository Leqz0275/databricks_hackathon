#!/bin/bash
exec streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.port=${DATABRICKS_APP_PORT:-8000} \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false
