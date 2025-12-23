FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN pip install mlflow scikit-learn pandas
CMD ["python", "modelling.py"]
