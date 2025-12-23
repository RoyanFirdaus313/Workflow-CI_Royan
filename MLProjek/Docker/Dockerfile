# Gunakan base image Python
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Salin file project
COPY ../MLProject /app

# Install dependencies
RUN pip install --upgrade pip
# Jika menggunakan conda.yaml
RUN pip install -r conda.yaml

# Jalankan script default
CMD ["python", "modelling.py"]
