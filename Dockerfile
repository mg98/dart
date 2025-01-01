FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
COPY allRank/requirements.txt requirements-allrank.txt
RUN pip install -e allRank
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["python", "main.py"]