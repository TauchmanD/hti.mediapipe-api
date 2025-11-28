FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV/mediapipe (minimal set; extend if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and model
COPY main.py .
COPY pose.task ./pose.task

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

