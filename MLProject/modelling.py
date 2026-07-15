import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

# Load data hasil preprocessing
X_train = pd.read_csv('namadataset_preprocessing/X_train.csv')
X_test = pd.read_csv('namadataset_preprocessing/X_test.csv')
y_train = pd.read_csv('namadataset_preprocessing/y_train.csv').values.ravel()
y_test = pd.read_csv('namadataset_preprocessing/y_test.csv').values.ravel()

# Set MLflow tracking ke lokal
# baris tracking uri dinonaktifkan untuk CI
# baris set_experiment dinonaktifkan untuk CI

# Aktifkan autolog (otomatis catat parameter & metrik)
mlflow.sklearn.autolog()

with mlflow.start_run():
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Akurasi model: {accuracy:.4f}")
