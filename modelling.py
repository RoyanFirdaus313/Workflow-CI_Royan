import argparse
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--l1_ratio", type=float, default=0.5)
args = parser.parse_args()

data = pd.read_csv("MLProject/namadataset_preprocessing/data.csv")
X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio)
model.fit(X_train, y_train)
preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)

mlflow.sklearn.log_model(model, "model")
mlflow.log_metric("rmse", rmse)
print(f"Model trained. RMSE: {rmse}")
