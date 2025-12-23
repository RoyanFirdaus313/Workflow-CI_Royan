# Base image Miniconda
FROM continuumio/miniconda3:latest

# Working directory
WORKDIR /app

# Copy project files
COPY . /app

# Create conda environment
RUN conda env create -f conda.yml

# Activate environment by default
SHELL ["conda", "run", "-n", "sml-env", "/bin/bash", "-c"]

# Default command
CMD ["python", "modelling.py"]
