"""
PrivateMed — Full-stack launcher.
Serves static/index.html AND the FastAPI /train, /mia_audit, /tradeoff_curve endpoints.

Usage:
  pip install -r requirements.txt
  python main.py

Then open:  http://localhost:8000
The HTML page works standalone too — just open static/index.html directly in a browser.
The JS will silently fall back to the built-in client-side simulation if the backend is unavailable.
"""

import os, sys
import uvicorn
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Import the FastAPI app from api/app.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "api"))
from app import app  # noqa: E402

static_dir = os.path.join(os.path.dirname(__file__), "static")

# Serve static assets under /static/
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Catch-all: serve index.html for any unmatched route (SPA behaviour)
# NOTE: Must be registered AFTER all API routes in app.py.
@app.get("/app/{full_path:path}", include_in_schema=False)
def serve_spa(full_path: str = ""):
    index = os.path.join(static_dir, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return JSONResponse({"error": "index.html not found"}, status_code=404)

# Root → index.html
@app.get("/ui", include_in_schema=False)
def serve_root():
    index = os.path.join(static_dir, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return JSONResponse({"error": "index.html not found"}, status_code=404)

if __name__ == "__main__":
    print("=" * 60)
    print("  PrivateMed — Safe & Private Hospital AI")
    print("  API + UI server starting on http://localhost:8000")
    print("  Open the UI at:  http://localhost:8000/ui")
    print("  API docs at:     http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

