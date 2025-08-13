import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.features.build_features import Preprocess

MODEL_PATH = "models/model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"


def main():
    print("Train your model")

    file_path = r"data/external/1000_sample_data.parquet"
    df = pd.read_parquet(file_path)

    target = "data_compl_usg_local_m1"
    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocess = Preprocess()
    preprocess.build(X_train)
    X_train_transformed = preprocess.fit_transform(X_train, y_train)
    X_test_transformed = preprocess.transform(X_test)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_transformed, y_train)

    y_pred = model.predict(X_test_transformed)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.3f}")
    print(f"MSE: {mse:.3f}")
    print(f"R² Score: {r2:.3f}")

    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(PREPROCESSOR_PATH, "wb") as f:
        pickle.dump(preprocess.preprocessor, f)

    print(f"✅ Model saved to {MODEL_PATH}")
    print(f"✅ Preprocessor saved to {PREPROCESSOR_PATH}")


if __name__ == "__main__":
    main()
