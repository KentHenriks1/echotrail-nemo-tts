FROM runpod/pytorch:2.6.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN pip install --no-cache-dir \
    chatterbox-tts \
    runpod \
    fastapi \
    uvicorn

COPY handler.py .

CMD ["python", "-u", "handler.py"]
