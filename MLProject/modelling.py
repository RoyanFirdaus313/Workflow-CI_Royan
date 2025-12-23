import argparse
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--l1_ratio", type=float, default=0.1)
args = parser.parse_args()

# Load dataset
df = pd.read_csv("namadataset_preprocessing/data.csv")
X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow run
with mlflow.start_run():
    model = ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)

    mlflow.log_param("alpha", args.alpha)
    mlflow.log_param("l1_ratio", args.l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(model, "model")

    print(f"Run completed with RMSE: {rmse}")
