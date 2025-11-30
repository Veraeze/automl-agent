from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def train_models(X, y):
    print("\n Training models...\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = []

    # If target is numeric → Regression models
    if y.dtype != "object":
        models = [
            ("Linear Regression", LinearRegression()),
            ("Random Forest Regressor", RandomForestRegressor())
        ]
    else:
        models = [
            ("Logistic Regression", LogisticRegression(max_iter=1000)),
            ("Random Forest Classifier", RandomForestClassifier())
        ]

    trained_models = []

    for name, model in models:
        model.fit(X_train, y_train)
        trained_models.append((name, model, X_test, y_test))

        print(f" {name} trained")

    return trained_models