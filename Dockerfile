FROM rocm/pytorch:latest

USER root
# Install system-level dependencies
RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Copy requirements first (Best practice for Docker caching)
COPY src/requirements.txt .

# 2. Install your specific requirements + Web dependencies
# We combine these to ensure all dependencies are resolved together
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
    git+https://github.com/m-bain/whisperX.git \
    fastapi \
    uvicorn \
    python-multipart \
    aiofiles

# 3. Copy the rest of your application code
COPY src .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
