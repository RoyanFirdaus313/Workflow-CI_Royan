# Gunakan base image Miniconda3
FROM continuumio/miniconda3:latest

# Set working directory di container
WORKDIR /app

# Salin semua isi folder MLProjek ke container
COPY . /app

# Buat environment conda dari conda.yml
RUN conda env create -f conda.yml

# Aktifkan environment conda secara default
SHELL ["conda", "run", "-n", "sml-env", "/bin/bash", "-c"]

# Default command saat container dijalankan
CMD ["python", "modelling.py"]


