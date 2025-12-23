import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Set experiment
mlflow.set_experiment("Workflow_CI_Royan")

# Gunakan path relatif yang aman
base_path = os.path.dirname(__file__)
dataset_path = os.path.join(base_path, "namadataset_preprocessing", "data.csv")

if not os.path.exists(dataset_path):
    print(f"Directory content: {os.listdir(base_path)}")
    raise FileNotFoundError(f"Dataset tidak ditemukan di {dataset_path}")

df = pd.read_csv(dataset_path)

# Asumsi kolom target bernama 'target'
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", acc)
    
    # Simpan artifact
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/model.pkl")
    mlflow.log_artifact("artifacts/model.pkl")
    
    print(f"Successfully logged run with accuracy: {acc}")
