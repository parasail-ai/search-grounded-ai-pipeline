FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV PORT=8091
ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

EXPOSE 8091

CMD ["python3", "server.py"]
