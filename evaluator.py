from sklearn.metrics import accuracy_score, r2_score

def evaluate_models(trained_models):
    print("\n Evaluating models...\n")

    results = []

    for name, model, X_test, y_test in trained_models:

        if y_test.dtype != "object":
            predictions = model.predict(X_test)
            score = r2_score(y_test, predictions)
            metric = "R² Score"
        else:
            predictions = model.predict(X_test)
            score = accuracy_score(y_test, predictions)
            metric = "Accuracy"

        print(f" {name} {metric}: {round(score, 4)}")
        results.append((name, model, score))
    
    return results

   