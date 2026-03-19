import pandas as pd
import xgboost as xgb
import os
import sys


def get_latest_team_stats(df: pd.DataFrame, team_abbr: str):
    """Finds the most recent game for a team to get their current 'Snapshot'."""
    team_data = df[df["TEAM_ABBREVIATION"] == team_abbr].sort_values(by="GAME_DATE")
    if team_data.empty:
        return None
    # Return the very last row (their most recent state)
    return team_data.iloc[-1:]


def predict_matchup(team_a: str, team_b: str) -> None:
    print(f"\nDivineLines: {team_a} vs {team_b}")
    print("-" * 40)

    # 1. Load the Data and the Model
    data_path = os.path.join("..", "data", "processed", "engineered_features.csv")
    model_path = os.path.join("..", "data", "processed", "divinelines_v1.json")

    if not os.path.exists(model_path):
        print("[ERROR] Could not find the saved model. Run train_model.py first!")
        return

    df = pd.read_csv(data_path)

    # Use XGBoost Booster class to load the model, since we saved it in JSON format
    model = xgb.Booster()
    model.load_model(model_path)

    # 2. Get the most recent stats for both teams
    team_a_stats = get_latest_team_stats(df, team_a)
    team_b_stats = get_latest_team_stats(df, team_b)

    if team_a_stats is None or team_b_stats is None:
        print(
            "[ERROR] One or both teams not found in the database. Check abbreviations (e.g., 'BOS', 'LAL')."
        )
        return

    # 3. Filter only the features the model was trained on
    features = [
        col
        for col in df.columns
        if col.startswith("ROLL_") or col.startswith("TREND_") or col == "DAYS_REST"
    ]

    X_team_a = team_a_stats[features]
    X_team_b = team_b_stats[features]

    # Pandas DataFrames into XGBoost's native DMatrix format
    dmatrix_a = xgb.DMatrix(X_team_a)
    dmatrix_b = xgb.DMatrix(X_team_b)

    # 4. Win Probabilities
    # The native Booster returns a single float (probability of class 1 / Win)
    prob_a = model.predict(dmatrix_a)[0]
    prob_b = model.predict(dmatrix_b)[0]

    # 5. Normalize the probabilities for a Head-to-Head matchup
    total_prob = prob_a + prob_b
    matchup_prob_a = (prob_a / total_prob) * 100
    matchup_prob_b = (prob_b / total_prob) * 100

    # 6. Display
    print(f"{team_a} Raw Win Probability: {prob_a * 100:.1f}%")
    print(f"{team_b} Raw Win Probability: {prob_b * 100:.1f}%")
    print("-" * 40)

    if matchup_prob_a > matchup_prob_b:
        print(f"PREDICTION: {team_a} wins with {matchup_prob_a:.1f}% confidence.")
    else:
        print(f"PREDICTION: {team_b} wins with {matchup_prob_b:.1f}% confidence.")
    print("-" * 40)


if __name__ == "__main__":
    # Test for now
    TEAM_1 = "BOS"
    TEAM_2 = "DAL"

    predict_matchup(TEAM_1, TEAM_2)
