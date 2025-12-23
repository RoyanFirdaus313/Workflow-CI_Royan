import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# Parsing argumen
parser = argparse.ArgumentParser()
parser.add_argument("--C", type=float, default=1.0)
args = parser.parse_args()

# Load dataset
data = pd.read_csv("MLProject/namadataset_preprocessing/data.csv")

# Pisahkan fitur dan target
X = data.drop("Berlangganan Deposito", axis=1)
y = data["Berlangganan Deposito"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training model
model = LogisticRegression(C=args.C, max_iter=1000)
model.fit(X_train, y_train)

# Prediksi dan evaluasi
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

# Logging ke MLflow
mlflow.sklearn.log_model(model, "model")
mlflow.log_metric("accuracy", acc)
print(f"Model trained. Accuracy: {acc}")
