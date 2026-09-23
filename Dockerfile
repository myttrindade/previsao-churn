FROM python:3.12-slim

# LightGBM precisa da biblioteca OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 app
WORKDIR /home/app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY --chown=app src/ src/
COPY --chown=app api/ api/
COPY --chown=app models/ models/

USER app
EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
