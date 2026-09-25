"""
api/index.py
------------
Serverless WSGI entry point for Vercel deployment.
Routes incoming serverless HTTP requests to the Flask application.
"""

import os
import sys

# Add project root directory to Python path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.app import app

# Vercel Python runtime detects and calls 'app'
