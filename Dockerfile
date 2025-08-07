FROM python:3.10-slim

RUN apt-get update && apt-get install -y git libgl1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY app/ /app

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

CMD ["python", "main.py"]
