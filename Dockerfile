FROM python:3.11-slim

WORKDIR /app

COPY . .

CMD ["python", "-c", "print('Docker image stub for disaster-tweet-classifier')"]
