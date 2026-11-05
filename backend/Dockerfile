FROM python:3.11-slim-bookworm
# Pinned to bookworm: the unpinned python:3.11-slim tag moves with Debian releases, and the
# package names below differ between them (libgl1-mesa-glx no longer exists at all).

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Model weights (GLiNER, sentence-transformers, EasyOCR) download on first use. Keep them
    # in the mounted data volume so a restart doesn't re-download them.
    HF_HOME=/app/data/hf \
    EASYOCR_MODULE_PATH=/app/data/easyocr

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch first: the default PyPI wheel bundles CUDA libraries (several GB) that a
# CPU server never uses. requirements.txt's torch==2.3.0 is then already satisfied.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.3.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# .dockerignore keeps .env, databases, data/ and local venvs out of the image.
COPY . .

RUN mkdir -p data/chroma data/hf data/easyocr

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# Behind nginx every request arrives from the proxy's address. --proxy-headers makes uvicorn
# use the client IP nginx forwards (login throttling and the audit log depend on it).
# "*" is safe ONLY because port 8000 is not published to the host (see docker-compose.yml)
# and nginx overwrites X-Forwarded-For. If you publish 8000, set this to nginx's address.
ENV FORWARDED_ALLOW_IPS="*"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
