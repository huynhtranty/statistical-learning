FROM python:3.11-slim

# OpenCV + Pillow native deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces requires non-root uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/home/user/app

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

ENV OD_MODEL=faster_rcnn \
    OD_DEVICE=cpu \
    OD_NUM_CLASSES=10

EXPOSE 7860
CMD ["uvicorn", "webapp.backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
