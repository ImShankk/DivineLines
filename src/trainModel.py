import pandas as pd
import xgboost as xgb
import os
from sklearn.metrics import accuracy_score
import itertools


def train_divinelines_model(data_path: str) -> None:

    # 1. Load the Data
    df = pd.read_csv(data_path)

    # 2. Define Features (The inputs) and Target (The answer key)
    features = [
        col for col in df.columns if col.startswith("DIFF_") or col == "H2H_WIN_PCT"
    ]

    X = df[features]
    y = df["HOME_WIN_TARGET"]

    # 3. Chronological Train/Test Split
    # 80/20 split to ensure we're training on the past and testing on the future
    train_idx = int(len(df) * 0.70)
    val_idx = int(len(df) * 0.85)

    X_train, y_train = X.iloc[:train_idx], y.iloc[:train_idx]
    X_val, y_val = X.iloc[train_idx:val_idx], y.iloc[train_idx:val_idx]
    X_test, y_test = X.iloc[val_idx:], y.iloc[val_idx:]

    print(f"-> Training on {len(X_train)} games...")
    print(f"-> Validating tuning on {len(X_val)} games...")
    print(f"-> Final Testing on {len(X_test)} future games...\n")

    # 4. Define the Grid & Generate Combinations
    param_grid = {
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "n_estimators": [100, 200, 300],
    }

    keys = param_grid.keys()
    combinations = list(itertools.product(*param_grid.values()))

    best_acc = 0
    best_params = {}
    best_model = None

    # 5. The Custom Grid Search Loop (should be workingnow?>)
    for combo in combinations:
        params = dict(zip(keys, combo))

        temp_model = xgb.XGBClassifier(**params, random_state=42)
        temp_model.fit(X_train, y_train)

        val_preds = temp_model.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)

        if val_acc > best_acc:
            best_acc = val_acc
            best_params = params
            best_model = temp_model

    # 6. Print the Winner
    print("====================================")
    print(" OPTIMAL HYPERPARAMETERS FOUND:")
    print(f" Max Depth:     {best_params['max_depth']}")
    print(f" Learning Rate: {best_params['learning_rate']}")
    print(f" N Estimators:  {best_params['n_estimators']}")
    print("====================================\n")

    # 7. Test the best model
    final_preds = best_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, final_preds)
    print(f"DivineLines V3 (Optimized) Accuracy: {final_accuracy * 100:.2f}%\n")

    # 8. What are the most important features
    importance = pd.DataFrame(
        {"Feature": features, "Importance": best_model.feature_importances_}
    ).sort_values(by="Importance", ascending=False)

    print("\n--- Most Important Features ---")
    print(importance.head(7).to_string(index=False))

    # Save the Optimized Brain
    model_path = os.path.join(
        "..", "data", "processed", "divinelines_v3_optimized.json"
    )
    best_model.save_model(model_path)
    print(f"\n[SUCCESS] Optimized Model saved to: {model_path}")
    print("The optimized AI is now ready to predict future games.")


if __name__ == "__main__":
    DATA = os.path.join("..", "data", "processed", "engineered_features.csv")
    train_divinelines_model(DATA)
