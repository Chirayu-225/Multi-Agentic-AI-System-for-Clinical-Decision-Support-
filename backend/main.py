"""
main.py — backend entry point
Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from src.api.main import app  # noqa: F401
