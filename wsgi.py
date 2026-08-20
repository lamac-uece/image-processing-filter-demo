"""WSGI entrypoint for PythonAnywhere (classic web apps).

See README deployment note: `demo.launch()` is guarded by __main__ in app.py,
so importing `demo` here is safe. a2wsgi adapts Gradio's ASGI/FastAPI app to
the WSGI interface PythonAnywhere serves. Requires: pip install a2wsgi
"""

from a2wsgi import ASGIMiddleware

from app import demo

application = ASGIMiddleware(demo.app)
