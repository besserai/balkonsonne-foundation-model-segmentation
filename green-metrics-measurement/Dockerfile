FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir "huggingface_hub[cli]"

WORKDIR /workspace/predict_only_without_app

CMD ["bash", "-lc", "tail -f /dev/null"]
