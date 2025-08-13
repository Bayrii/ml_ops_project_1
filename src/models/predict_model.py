import pickle

import pandas as pd

MODEL_PATH = "models/model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"


def predict(new_data_path):
    df = pd.read_parquet(new_data_path)

    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)
    X_transformed = preprocessor.transform(df)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    predictions = model.predict(X_transformed)

    return predictions


if __name__ == "__main__":

    print("Load model and run predictions I DID SOME CHANGES")

    preds = predict("data/processed/test_data.parquet")
    print("Predictions:", preds)
