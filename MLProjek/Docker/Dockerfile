# Gunakan base image Python
FROM python:3.11-slim

# Set working directory di container
WORKDIR /app

# Salin seluruh isi folder build context (MLProjek) ke container
COPY . /app

# Install dependencies pip
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


# Default command saat container dijalankan
CMD ["python", "modelling.py"]

