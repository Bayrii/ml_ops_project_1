import pandas as pd
from category_encoders import CatBoostEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


class SplitRowsByTypes:
    def __init__(self):
        self.numeric_features = []
        self.categorical_features = []
        self.low_cardinality_features = []
        self.high_cardinality_features = []
        self.threshold = 0

    def set_threshold(self, X_train, threshold_percentage=0.05):
        num_rows = len(X_train)
        self.threshold = num_rows * threshold_percentage

    def set_categorical_features(self, X_train):
        self.categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    def get_numeric_features(self, X_train):
        self.numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
        return self.numeric_features

    def get_low_cardinality_features(self, X_train):
        self.low_cardinality_features = [
            col for col in self.categorical_features if X_train[col].nunique() <= self.threshold
        ]
        return self.low_cardinality_features

    def get_high_cardinality_features(self, X_train):
        self.high_cardinality_features = [
            col for col in self.categorical_features if X_train[col].nunique() > self.threshold
        ]
        return self.high_cardinality_features


class Preprocess:
    def __init__(self, threshold_percentage=0.05):
        self.rows = SplitRowsByTypes()
        self.threshold_percentage = threshold_percentage
        self.numeric_transformer = None
        self.low_categorical_transformer = None
        self.high_categorical_transformer = None
        self.preprocessor = None

    def build(self, X_train):

        self.rows.set_threshold(X_train, self.threshold_percentage)
        self.rows.set_categorical_features(X_train)
        numeric_features = self.rows.get_numeric_features(X_train)
        low_cat_features = self.rows.get_low_cardinality_features(X_train)
        high_cat_features = self.rows.get_high_cardinality_features(X_train)

        self.numeric_transformer = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]
        )

        self.low_categorical_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        self.high_categorical_transformer = Pipeline(
            [("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", CatBoostEncoder())]
        )

        self.preprocessor = ColumnTransformer(
            [
                ("num", self.numeric_transformer, numeric_features),
                ("low_cat", self.low_categorical_transformer, low_cat_features),
                ("high_cat", self.high_categorical_transformer, high_cat_features),
            ]
        )

        return self

    def fit_transform(self, X_train, y=None):
        if y is not None:
            return self.preprocessor.fit_transform(X_train, y)
        return self.preprocessor.fit_transform(X_train)

    def transform(self, X):
        return self.preprocessor.transform(X)


def main():
    print("Transform raw data into features")
    file_path = r"data/external/1000_sample_data.parquet"
    data = pd.read_parquet(file_path)

    preprocess = Preprocess()
    preprocess.build(data)
    transformed_data = preprocess.fit_transform(data)

    print("Shape after transformation:", transformed_data.shape)


if __name__ == "__main__":
    main()
