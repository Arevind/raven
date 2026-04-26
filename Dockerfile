FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    libsndfile1 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt \
    && pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

COPY voicebot ./voicebot
COPY scripts ./scripts
COPY tests ./tests
COPY docker/entrypoint.sh /usr/local/bin/voicebot-entrypoint

RUN chmod +x /usr/local/bin/voicebot-entrypoint \
    && useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /cache/huggingface \
    && chown -R appuser:appuser /app /cache/huggingface

USER appuser

EXPOSE 7860

ENTRYPOINT ["voicebot-entrypoint"]
CMD ["python", "-m", "voicebot.cli.dashboard", "--host", "0.0.0.0", "--port", "7860"]
