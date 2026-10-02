FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN useradd -m -u 1000 user
WORKDIR /app
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements-deploy.txt .
RUN pip install -r requirements-deploy.txt
COPY . .
RUN chown -R user:user /app
USER user
ENV MODEL_PATH=/app/backend/models_store/damage_classifier.pt
EXPOSE 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]