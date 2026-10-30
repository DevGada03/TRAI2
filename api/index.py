"""
Vercel Serverless Function entry point.
Exposes FastAPI application via 'app' for Vercel Python runtime.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from app import app
