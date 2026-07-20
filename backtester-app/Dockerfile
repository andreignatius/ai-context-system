FROM python:3.11-slim
# stream logs unbuffered; skip .pyc files (smaller, cleaner image)
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# deps FIRST, in their own layer — they change rarely, so Docker caches this
# and skips re-installing every time you edit source. (Order matters for caching.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# then the code
COPY src/ ./src/

EXPOSE 8501

# ui.py adds /app to sys.path, so `from src.graph import ...` works.
# --server.address=0.0.0.0 makes the port reachable from OUTSIDE the container
# (default 'localhost' inside a container isn't visible on your host).
CMD ["streamlit", "run", "src/ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
