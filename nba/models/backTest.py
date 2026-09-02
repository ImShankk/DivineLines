import sqlite3
import pandas as pd
import xgboost as xgb
import os


def run_backtest(confidence_threshold=55.0, wager_amount=60):
    print(f"--- INITIATING QUANTITATIVE BACKTEST ---")
    print(f"Confidence Threshold: {confidence_threshold}%")
    print(f"Unit Size: ${wager_amount}\n")

    # load the optimized xgb model
    model_path = os.path.join(
        "..", "data", "processed", "divinelines_v3_optimized.json"
    )
    if not os.path.exists(model_path):
        print("[!] Error: Model not found. Run training first.")
        return

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    expected_cols = model.get_booster().feature_names

    # pull historicals from the db
    db_path = os.path.join("..", "data", "processed", "nba_data.db")

    try:
        conn = sqlite3.connect(db_path)
        # only grab games that have actually finished
        query = "SELECT * FROM historical_features WHERE ACTUAL_WINNER IS NOT NULL"
        df = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as e:
        print(
            "[!] Note: You need to generate a 'historical_features' table to run a full season backtest."
        )
        return

    # pad missing columns with 0 so the model doesn't freak out
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    X_test = df[expected_cols]
    actual_results = df["ACTUAL_WINNER"]

    # batch predict the whole season
    probabilities = model.predict_proba(X_test)

    # run the simulation
    total_bets = 0
    wins = 0
    losses = 0
    profit = 0.0

    # hardcoded standard -110 juice for now
    profit_per_win = wager_amount * (100 / 110)
    loss_amount = wager_amount

    for i in range(len(probabilities)):
        home_prob = probabilities[i][1] * 100
        away_prob = probabilities[i][0] * 100
        actual_winner = actual_results.iloc[i]

        # only trigger bet if we cross the confidence threshold
        if home_prob >= confidence_threshold:
            total_bets += 1
            if actual_winner == 1:
                wins += 1
                profit += profit_per_win
            else:
                losses += 1
                profit -= loss_amount

        elif away_prob >= confidence_threshold:
            total_bets += 1
            if actual_winner == 0:
                wins += 1
                profit += profit_per_win
            else:
                losses += 1
                profit -= loss_amount

    # tally up roi metrics
    if total_bets == 0:
        print("No bets placed. Model did not reach confidence threshold on any games.")
        return

    win_rate = (wins / total_bets) * 100
    total_wagered = total_bets * wager_amount
    roi = (profit / total_wagered) * 100

    print("=========================================")
    print("           BACKTEST RESULTS              ")
    print("=========================================")
    print(f"Total Games Analyzed: {len(df)}")
    print(f"Total Bets Placed:    {total_bets}")
    print(f"Win/Loss Record:      {wins} - {losses}")
    print(f"Win Rate:             {win_rate:.2f}%")
    print("-----------------------------------------")
    print(f"Total Wagered:        ${total_wagered:,.2f}")
    if profit > 0:
        print(f"Net Profit:           +${profit:,.2f}")
    else:
        print(f"Net Profit:           -${abs(profit):,.2f}")
    print(f"ROI:                  {roi:.2f}%")
    print("=========================================")


if __name__ == "__main__":
    # override default threshold here for testing
    run_backtest(confidence_threshold=58.0)
