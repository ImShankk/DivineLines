import pandas as pd
import xgboost as xgb
import os
from sklearn.metrics import accuracy_score, classification_report


def train_divinelines_model(data_path: str) -> None:

    # 1. Load the Data
    df = pd.read_csv(data_path)

    # 2. Define Features (The inputs) and Target (The answer key)
    features = [col for col in df.columns if col.startswith("DIFF_")]

    X = df[features]
    y = df["HOME_WIN_TARGET"]

    # 3. Chronological Train/Test Split
    # 80/20 split to ensure we're training on the past and testing on the future
    split_index = int(len(df) * 0.80)

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f"-> Training on {len(X_train)} historical games...")
    print(f"-> Testing on {len(X_test)} unseen future games...\n")

    # 4. Initialize the XGBoost Classifier
    model = xgb.XGBClassifier(
        n_estimators=150,  # Number of decision trees (changed from 100)
        learning_rate=0.05,  # How fast the model learns (changed from 0.1)
        max_depth=5,  # How deep the trees go (prevents overthinking) (changed from 4)
        random_state=42,  # Ensures we get the same results every time we run it
    )

    # 5. Train
    model.fit(X_train, y_train)

    # 6. Predict on the Test Set
    predictions = model.predict(X_test)

    # 7. Test Accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"DivineLines V1 Accuracy: {accuracy * 100:.2f}%")

    # 8. What are the most important features?
    # Most likely to influence the model's predictions (e.g., recent momentum, points scored, etc.)
    importance = pd.DataFrame(
        {"Feature": features, "Importance": model.feature_importances_}
    ).sort_values(by="Importance", ascending=False)

    print("--- Top 5 Most Important Features ---")
    print(importance.to_string(index=False))

    # Save this so I dont have to retrain the model everytime
    model_path = os.path.join("..", "data", "processed", "divinelines_v2.json")
    model.save_model(model_path)
    print(f"\n[SUCCESS] Model saved to: {model_path}")
    print("The AI is now ready to predict future games.")  # (FIRST VERSION 61.43%)


if __name__ == "__main__":
    DATA = os.path.join("..", "data", "processed", "engineered_features.csv")
    train_divinelines_model(DATA)
