import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import shutil

# 1. Konfigurasi Eksperimen
mlflow.set_experiment("Workflow_CI_Royan")

# 2. Load Dataset (Pastikan path folder benar di repo Anda)
dataset_path = os.path.join("namadataset_preprocessing", "data.csv")

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset tidak ditemukan di: {dataset_path}")

df = pd.read_csv(dataset_path)

# 3. Preprocessing Dasar
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Training & Tracking
with mlflow.start_run():
    # Model Parameter
    n_estimators = 50
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    # Evaluasi
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    # Log ke MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", acc)
    
    # Simpan Model sebagai Artifact MLflow Resmi
    mlflow.sklearn.log_model(model, "random_forest_model")

    # 5. Simpan Local Artifact (untuk workflow Docker/Drive Anda)
    artifact_dir = "artifacts"
    if os.path.exists(artifact_dir):
        shutil.rmtree(artifact_dir) # Bersihkan folder lama
    os.makedirs(artifact_dir, exist_ok=True)
    
    model_file = os.path.join(artifact_dir, "model.pkl")
    joblib.dump(model, model_file)
    mlflow.log_artifact(model_file)

    print(f"✅ Training Selesai. Accuracy: {acc:.4f}")
    print(f"📂 Model disimpan di: {model_file}")
