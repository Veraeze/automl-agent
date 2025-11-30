import joblib

def select_and_save_best_model(results):
    best_model = sorted(results, key=lambda x: x[2], reverse=True)[0]

    name, model, score = best_model

    joblib.dump(model, f"models/{name.replace(' ', '_')}.pkl")

    print(f"\n Best model: {name} (Score: {round(score, 4)})")
    print("Model saved to /models folder")