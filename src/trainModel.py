import pandas as pd
import xgboost as xgb
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV


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
    split_index = int(len(df) * 0.80)

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f"-> Training on {len(X_train)} historical games...")
    print(f"-> Testing on {len(X_test)} unseen future games...\n")

    # Thank you Green Code : https://www.youtube.com/watch?v=N4JDlSTMOck for the idea of tuning hyperparameters.
    # 4. Set up the Grid Search Parameters
    param_grid = {
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "n_estimators": [100, 200, 300],
    }

    # 5. Time-Series Cross Validation
    tscv = TimeSeriesSplit(n_splits=3)

    # 6. Initialize the Tuner
    base_model = xgb.XGBClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=tscv,
        scoring="accuracy",
        verbose=1,
    )

    # 7. Execute the Grid Search on the Training Data
    grid_search.fit(X_train, y_train)

    # 8. Extract the Best Model
    best_model = grid_search.best_estimator_

    print("\n====================================")
    print(" OPTIMAL HYPERPARAMETERS FOUND:")
    print(f" Max Depth:     {grid_search.best_params_['max_depth']}")
    print(f" Learning Rate: {grid_search.best_params_['learning_rate']}")
    print(f" N Estimators:  {grid_search.best_params_['n_estimators']}")
    print("====================================\n")

    # 9. Test the optimized model on the test 20%
    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"DivineLines V3 (Optimized) Accuracy: {accuracy * 100:.2f}%\n")

    # 10. Feature Importance
    importance = pd.DataFrame(
        {"Feature": features, "Importance": best_model.feature_importances_}
    ).sort_values(by="Importance", ascending=False)

    print("--- Top 7 Most Important Features ---")
    print(importance.head(7).to_string(index=False))

    # 11. Save the Optimized Brain
    model_path = os.path.join(
        "..", "data", "processed", "divinelines_v3_optimized.json"
    )
    best_model.save_model(model_path)
    print(f"\n[SUCCESS] Optimized Model saved to: {model_path}")


if __name__ == "__main__":
    DATA = os.path.join("..", "data", "processed", "engineered_features.csv")
    train_divinelines_model(DATA)
