import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Konfigurasi Path dan Eksperimen
mlflow.set_experiment("Eksperimen_SML_Royan_Basic")

# Pastikan path dataset relatif terhadap lokasi script
base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, "namadataset_preprocessing", "data.csv")

# 2. Load data
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset tidak ditemukan di {data_path}")

df = pd.read_csv(data_path)

# 3. Pemisahan Fitur dan Target (Sesuai kolom dataset Anda)
# Pastikan kolom 'berlangganan_deposito' ada di file CSV
X = df.drop("berlangganan_deposito", axis=1)
y = df["berlangganan_deposito"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Training dengan MLflow Autolog
with mlflow.start_run():
    # autolog() akan otomatis mencatat parameter model, metrik, dan artifact
    mlflow.autolog()

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"✅ Berhasil! Accuracy: {acc}")
    
    # Opsional: Simpan model secara manual jika diperlukan di folder artifacts
    os.makedirs("artifacts", exist_ok=True)
    mlflow.sklearn.log_model(model, "random_forest_model")
