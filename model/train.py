import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.data.load_data import load_data
from src.features.build_features import engineer_features


def train():
    df = load_data("data/Titanic-Dataset.csv")
    df = engineer_features(df)

    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # param_grid = {
    # "model__C": [0.01, 0.1, 1, 10],
    # "model__class_weight": [
    #     "balanced",
    #     {0:1, 1:2},
    #     {0:1, 1:3}
    # ]
    # }

    # grid = GridSearchCV(
    # model,
    # param_grid,
    # cv=5,
    # scoring="recall",
    # n_jobs=-1
    # )

    # grid.fit(X_train, y_train)

    # best_model = grid.best_estimator_

    
    model.fit(X_train,y_train)

    joblib.dump(model, "model/titanic_logreg.pkl")

    print("Model saved successfully!")



if __name__ == "__main__":
    train()