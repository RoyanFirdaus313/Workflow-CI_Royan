import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# HAPUS ATAU KOMENTAR BARIS INI:
# mlflow.set_experiment("Eksperimen_SML_Royan_Basic") 

# --- Cari Dataset ---
possible_paths = [
    "namadataset_preprocessing/data.csv",
    "MLProject/namadataset_preprocessing/data.csv",
    "data.csv"
]

data_path = None
for p in possible_paths:
    if os.path.exists(p):
        data_path = p
        break

if data_path is None:
    print("❌ Dataset tidak ditemukan. Isi folder:")
    os.system("ls -R")
    raise FileNotFoundError("Gagal menemukan cleaned_df.csv")

df = pd.read_csv(data_path)

# --- Training ---
X = df.drop("berlangganan_deposito", axis=1)
y = df["berlangganan_deposito"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gunakan start_run() tanpa argumen karena sudah diatur oleh MLflow CLI
with mlflow.start_run():
    mlflow.autolog()
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Berhasil! Accuracy: {acc}")
