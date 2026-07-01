FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        torch==2.4.0 \
        torchvision==0.19.0 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir \
        transformers \
        pillow \
        numpy \
        "huggingface_hub[cli]"

WORKDIR /workspace/predict_only_without_app

CMD ["bash", "-lc", "tail -f /dev/null"]
