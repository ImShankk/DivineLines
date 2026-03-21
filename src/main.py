import sqlite3
import pandas as pd
import xgboost as xgb
import os
import time
import sys


try:
    from syncData import sync_database
except ImportError:
    pass


def get_latest_team_stats(db_path: str, team_abbr: str):
    """Calculates the true current rolling averages for a team based on their last 10 games."""
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM game_logs WHERE TEAM_ABBREVIATION = '{team_abbr}' ORDER BY GAME_DATE ASC"
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        return None

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["WIN_BIN"] = df["WL"].apply(lambda x: 1 if x == "W" else 0)

    # Grab the absolute last 10 games for current form
    last_10 = df.tail(10)

    stats = {}

    stats["ROLL_10_PTS"] = last_10["PTS"].mean()
    stats["ROLL_10_PLUS_MINUS"] = last_10["PLUS_MINUS"].mean()
    stats["ROLL_10_FG_PCT"] = last_10["FG_PCT"].mean()
    stats["ROLL_10_FG3_PCT"] = last_10["FG3_PCT"].mean()
    stats["ROLL_10_FT_PCT"] = last_10["FT_PCT"].mean()
    stats["ROLL_10_REB"] = last_10["REB"].mean()
    stats["ROLL_10_OREB"] = last_10["OREB"].mean()
    stats["ROLL_10_AST"] = last_10["AST"].mean()
    stats["ROLL_10_TOV"] = last_10["TOV"].mean()
    stats["ROLL_10_STL"] = last_10["STL"].mean()
    stats["ROLL_10_BLK"] = last_10["BLK"].mean()
    stats["ROLL_10_PF"] = last_10["PF"].mean()
    stats["ROLL_10_WIN_PCT"] = last_10["WIN_BIN"].mean()

    # Calculate exact Days Rest
    last_game_date = df.iloc[-1]["GAME_DATE"]
    days_since_last_game = (pd.Timestamp.today() - last_game_date).days
    stats["DAYS_REST"] = min(max(days_since_last_game, 0), 4)

    return stats


def get_h2h_win_pct(db_path: str, home_team: str, away_team: str):
    """Calculates the historical head-to-head win percentage for the home team against the away team."""
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT WL FROM game_logs 
        WHERE TEAM_ABBREVIATION = '{home_team}' 
        AND MATCHUP LIKE '%{away_team}%'
        AND GAME_DATE >= '2025-10-01'
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        return 0.50
    return len(df[df["WL"] == "W"]) / len(df)


def generate_explanation(home, away, diffs, home_prob):
    """Translates the mathematical differentials into a plain-English explanation."""
    favored = home if home_prob > 50 else away

    reasons = []

    # If Home is favored, look for positive differentials. If Away is favored, look for negative.
    if home_prob > 50:
        if diffs["DIFF_ROLL_10_PLUS_MINUS"] > 3.0:
            reasons.append(
                f"a vastly superior Net Rating (+{diffs['DIFF_ROLL_10_PLUS_MINUS']:.1f})"
            )
        elif diffs["DIFF_ROLL_10_PLUS_MINUS"] > 0.5:
            reasons.append(
                f"a slight Net Rating edge (+{diffs['DIFF_ROLL_10_PLUS_MINUS']:.1f})"
            )

        if diffs["DIFF_DAYS_REST"] > 0:
            reasons.append(f"a {int(diffs['DIFF_DAYS_REST'])}-day rest advantage")

        if diffs["DIFF_ROLL_10_REB"] > 2.0:
            reasons.append(
                f"dominance on the glass (+{diffs['DIFF_ROLL_10_REB']:.1f} reb/game)"
            )
    else:
        if diffs["DIFF_ROLL_10_PLUS_MINUS"] < -3.0:
            reasons.append(
                f"a vastly superior Net Rating (+{abs(diffs['DIFF_ROLL_10_PLUS_MINUS']):.1f})"
            )
        elif diffs["DIFF_ROLL_10_PLUS_MINUS"] < -0.5:
            reasons.append(
                f"a slight Net Rating edge (+{abs(diffs['DIFF_ROLL_10_PLUS_MINUS']):.1f})"
            )

        if diffs["DIFF_DAYS_REST"] < 0:
            reasons.append(f"a {abs(int(diffs['DIFF_DAYS_REST']))}-day rest advantage")

        if diffs["DIFF_ROLL_10_REB"] < -2.0:
            reasons.append(
                f"dominance on the glass (+{abs(diffs['DIFF_ROLL_10_REB']):.1f} reb/game)"
            )

    if not reasons:
        return f"The matchup is quite close, but {favored} have the advantage based on the data."

    return (
        f"The winning team will likely be {favored} primarily due to "
        + " and ".join(reasons)
        + "."
    )


def run_oracle():
    db_path = os.path.join("..", "data", "processed", "nba_data.db")
    model_path = os.path.join(
        "..", "data", "processed", "divinelines_v3_optimized.json"
    )

    if not os.path.exists(model_path):
        print("[ERROR] Model not found. Run train_model.py first")
        input("\nPress Enter to return to menu...")
        return

    print("Loading DivineLines")
    model = xgb.Booster()
    model.load_model(model_path)

    # Clear the terminal screen for a clean UI every time we run it
    os.system("cls" if os.name == "nt" else "clear")

    print("=====================================================")
    print("                 DIVINELINES V3.0                    ")
    print("=====================================================")
    print("Type 'exit' or 'quit' at any time to close the app.\n")

    while True:
        away_team = input("Enter AWAY Team (e.g., DAL): ").strip().upper()
        if away_team in ["EXIT", "QUIT"]:
            break

        home_team = input("Enter HOME Team (e.g., BOS): ").strip().upper()
        if home_team in ["EXIT", "QUIT"]:
            break

        print("\nChecking... \n")
        time.sleep(
            0.4
        )  # Just a little dramatic pause for effect (dont need it but thought it would be funny)

        home_stats = get_latest_team_stats(db_path, home_team)
        away_stats = get_latest_team_stats(db_path, away_team)
        h2h_win_pct = get_h2h_win_pct(db_path, home_team, away_team)

        if home_stats is None or away_stats is None:
            print(
                f"[!] Could not find data for {away_team} or {home_team}. Check spelling.\n"
            )
            continue

        # The V2 Alpha: Calculate the Differentials
        diff_features = {
            "DIFF_ROLL_10_PTS": home_stats["ROLL_10_PTS"] - away_stats["ROLL_10_PTS"],
            "DIFF_ROLL_10_PLUS_MINUS": home_stats["ROLL_10_PLUS_MINUS"]
            - away_stats["ROLL_10_PLUS_MINUS"],
            "DIFF_ROLL_10_FG_PCT": home_stats["ROLL_10_FG_PCT"]
            - away_stats["ROLL_10_FG_PCT"],
            "DIFF_ROLL_10_FG3_PCT": home_stats["ROLL_10_FG3_PCT"]
            - away_stats["ROLL_10_FG3_PCT"],
            "DIFF_ROLL_10_FT_PCT": home_stats["ROLL_10_FT_PCT"]
            - away_stats["ROLL_10_FT_PCT"],
            "DIFF_ROLL_10_REB": home_stats["ROLL_10_REB"] - away_stats["ROLL_10_REB"],
            "DIFF_ROLL_10_OREB": home_stats["ROLL_10_OREB"]
            - away_stats["ROLL_10_OREB"],
            "DIFF_ROLL_10_AST": home_stats["ROLL_10_AST"] - away_stats["ROLL_10_AST"],
            "DIFF_ROLL_10_TOV": home_stats["ROLL_10_TOV"] - away_stats["ROLL_10_TOV"],
            "DIFF_ROLL_10_STL": home_stats["ROLL_10_STL"] - away_stats["ROLL_10_STL"],
            "DIFF_ROLL_10_BLK": home_stats["ROLL_10_BLK"] - away_stats["ROLL_10_BLK"],
            "DIFF_ROLL_10_PF": home_stats["ROLL_10_PF"] - away_stats["ROLL_10_PF"],
            "DIFF_ROLL_10_WIN_PCT": home_stats["ROLL_10_WIN_PCT"]
            - away_stats["ROLL_10_WIN_PCT"],
            "DIFF_DAYS_REST": home_stats["DAYS_REST"] - away_stats["DAYS_REST"],
            "H2H_WIN_PCT": h2h_win_pct,
        }

        matchup_df = pd.DataFrame([diff_features])
        dmatrix = xgb.DMatrix(matchup_df)

        home_win_prob = model.predict(dmatrix)[0] * 100
        away_win_prob = 100 - home_win_prob

        # Display the UI
        print("-" * 53)
        if home_win_prob > 50:
            print(
                f"PREDICTION: {home_team} (HOME) wins with {home_win_prob:.1f}% confidence."
            )
        else:
            print(
                f"PREDICTION: {away_team} (AWAY) wins with {away_win_prob:.1f}% confidence."
            )

        # Explain the reasoning
        explanation = generate_explanation(
            home_team, away_team, diff_features, home_win_prob
        )
        print(f" ANALYSIS:\n{explanation}")
        print("-" * 53 + "\n  \n")


def main_menu():
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print("=====================================================")
        print("                 DIVINELINES V3.0                    ")
        print("=====================================================")
        print(" [1] Sync Database (Fetch Last Night's Games)")
        print(" [2] Launch Predictor Oracle (V3 Optimized)")
        print(" [3] Exit")
        print("=====================================================")

        choice = input("\nSelect a command (1-3): ").strip()

        if choice == "1":
            os.system("cls" if os.name == "nt" else "clear")
            print("Initiating Database Sync...\n")
            try:
                sync_database()
                print("\n[SUCCESS] Database is up to date.")
            except NameError:
                print("[!] ERROR: Sync function not linked.")
                print(
                    "Open main.py and update the import statement at the top to point to your sync file."
                )
            input("\nPress Enter to return to menu...")

        elif choice == "2":
            run_oracle()

        elif choice == "3":
            os.system("cls" if os.name == "nt" else "clear")
            print("Shutting down DivineLines...")
            sys.exit(0)

        else:
            print("\n[!] Invalid selection. Please enter 1, 2, or 3.")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main_menu()
