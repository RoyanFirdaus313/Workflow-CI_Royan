# Base image Miniconda
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# 1️⃣ Salin conda.yml saja dulu → memungkinkan caching
COPY conda.yml /app/conda.yml

# 2️⃣ Buat environment conda
RUN conda env create -f conda.yml

# 3️⃣ Aktifkan environment
SHELL ["conda", "run", "-n", "sml-env", "/bin/bash", "-c"]

# 4️⃣ Salin sisa project
COPY . /app

# 5️⃣ Default command
CMD ["python", "modelling.py"]

