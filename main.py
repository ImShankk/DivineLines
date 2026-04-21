import sqlite3
import pandas as pd
import xgboost as xgb
import os
import time
import sys

try:
    from core.syncData import sync_database
except ImportError:
    pass


# restarting this page, becuase i think i broke it :0


def generate_explanation(home, away, diffs, home_prob):
    """Translates the mathematical differentials into a plain-English explanation."""
    favored = home if home_prob > 50 else away
    reasons = []

    if home_prob > 50:
        if diffs.get("DIFF_NET_RATING", 0) > 3.0:
            reasons.append(
                f"a vastly superior Net Rating (+{diffs.get('DIFF_NET_RATING', 0):.1f})"
            )
        elif diffs.get("DIFF_NET_RATING", 0) > 0.5:
            reasons.append(
                f"a slight Net Rating edge (+{diffs.get('DIFF_NET_RATING', 0):.1f})"
            )

        if diffs.get("HOME_3PT_ADVANTAGE", 0) > diffs.get("AWAY_3PT_ADVANTAGE", 0):
            reasons.append("a distinct advantage exploiting the perimeter defense")
    else:
        if diffs.get("DIFF_NET_RATING", 0) < -3.0:
            reasons.append(
                f"a vastly superior Net Rating (+{abs(diffs.get('DIFF_NET_RATING', 0)):.1f})"
            )
        elif diffs.get("DIFF_NET_RATING", 0) < -0.5:
            reasons.append(
                f"a slight Net Rating edge (+{abs(diffs.get('DIFF_NET_RATING', 0)):.1f})"
            )

        if diffs.get("AWAY_3PT_ADVANTAGE", 0) > diffs.get("HOME_3PT_ADVANTAGE", 0):
            reasons.append("a distinct advantage exploiting the perimeter defense")

    if not reasons:
        return f"The matchup is quite close mathematically, but {favored} holds the historical data advantage."

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

    os.system("cls" if os.name == "nt" else "clear")

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at: {model_path}")
        input("\nPress Enter to return to menu...")
        return

    print("Loading DivineLines V4 Brain...")
    try:
        model = xgb.XGBClassifier()
        model.load_model(model_path)
    except Exception as e:
        print(f"\n[!] CRITICAL ERROR: Could not load the XGBoost model.")
        print(f"Details: {e}")
        input("\nPress Enter to return to menu...")
        return

    print(
        "Loading Database and calculating V4 metrics... (This may take a few seconds)"
    )
    try:
        conn = sqlite3.connect(db_path)
        query = "SELECT * FROM game_logs ORDER BY GAME_DATE ASC"
        raw_df = pd.read_sql_query(query, conn)
        conn.close()

        raw_df["GAME_DATE"] = pd.to_datetime(raw_df["GAME_DATE"])
        raw_df.sort_values(by=["GAME_ID", "TEAM_ID"], inplace=True)

        raw_df["POSS"] = (
            raw_df["FGA"] - raw_df["OREB"] + raw_df["TOV"] + (0.44 * raw_df["FTA"])
        )
        raw_df["ORTG"] = (raw_df["PTS"] / raw_df["POSS"]) * 100
        raw_df["POINT_DIFF"] = raw_df["PLUS_MINUS"]

        raw_df["OPP_PTS"] = raw_df.groupby("GAME_ID")["PTS"].transform(
            lambda x: x.iloc[::-1].values
        )
        raw_df["OPP_POSS"] = raw_df.groupby("GAME_ID")["POSS"].transform(
            lambda x: x.iloc[::-1].values
        )
        raw_df["OPP_FG3A"] = raw_df.groupby("GAME_ID")["FG3A"].transform(
            lambda x: x.iloc[::-1].values
        )
        raw_df["OPP_FG3M"] = raw_df.groupby("GAME_ID")["FG3M"].transform(
            lambda x: x.iloc[::-1].values
        )
        raw_df["OPP_FTA"] = raw_df.groupby("GAME_ID")["FTA"].transform(
            lambda x: x.iloc[::-1].values
        )

        raw_df["DRTG"] = (raw_df["OPP_PTS"] / raw_df["OPP_POSS"]) * 100
        raw_df["NET_RATING"] = raw_df["ORTG"] - raw_df["DRTG"]
        raw_df["OPP_FG3_PCT"] = raw_df.apply(
            lambda row: row["OPP_FG3M"] / row["OPP_FG3A"] if row["OPP_FG3A"] > 0 else 0,
            axis=1,
        )
        raw_df["WIN_BIN"] = raw_df["WL"].apply(lambda x: 1 if x == "W" else 0)
    except Exception as e:
        print(f"\n[!] CRITICAL ERROR: Failed to process database math.")
        print(f"Details: {e}")
        input("\nPress Enter to return to menu...")
        return

    def get_latest_stats(team_abbr):
        team_data = (
            raw_df[raw_df["TEAM_ABBREVIATION"] == team_abbr]
            .sort_values(by="GAME_DATE")
            .tail(10)
        )
        if team_data.empty:
            return None
        return team_data.mean(numeric_only=True)

    os.system("cls" if os.name == "nt" else "clear")
    print("=====================================================")
    print("                DIVINELINES V4.0                     ")
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
        time.sleep(0.4)

        home_stats = get_latest_stats(home_team)
        away_stats = get_latest_stats(away_team)

        if home_stats is None or away_stats is None:
            print(
                f"[!] Could not find data for {away_team} or {home_team}. Check spelling.\n"
            )
            continue

        features = {}
        features["DIFF_PACE"] = home_stats["POSS"] - away_stats["POSS"]
        features["DIFF_NET_RATING"] = (
            home_stats["NET_RATING"] - away_stats["NET_RATING"]
        )
        features["DIFF_POINT_MARGIN"] = (
            home_stats["POINT_DIFF"] - away_stats["POINT_DIFF"]
        )

        features["HOME_3PT_ADVANTAGE"] = home_stats["FG3A"] * away_stats["OPP_FG3_PCT"]
        features["AWAY_3PT_ADVANTAGE"] = away_stats["FG3A"] * home_stats["OPP_FG3_PCT"]
        features["HOME_FT_ADVANTAGE"] = home_stats["FTA"] - away_stats["OPP_FTA"]
        features["AWAY_FT_ADVANTAGE"] = away_stats["FTA"] - home_stats["OPP_FTA"]

        for stat in ["ORTG", "DRTG", "REB", "AST", "TOV", "WIN_BIN"]:
            features[f"DIFF_ROLL_10_{stat}"] = home_stats[stat] - away_stats[stat]

        features["H2H_WIN_PCT"] = 0.50

        pred_df = pd.DataFrame([features])

        expected_cols = model.get_booster().feature_names
        for col in expected_cols:
            if col not in pred_df.columns:
                pred_df[col] = 0
        pred_df = pred_df[expected_cols]

        home_win_prob = model.predict_proba(pred_df)[0][1] * 100
        away_win_prob = 100 - home_win_prob

        print("-" * 53)
        if home_win_prob > 50:
            print(
                f"PREDICTION: {home_team} (HOME) wins with {home_win_prob:.1f}% confidence."
            )
        else:
            print(
                f"PREDICTION: {away_team} (AWAY) wins with {away_win_prob:.1f}% confidence."
            )

        explanation = generate_explanation(
            home_team, away_team, features, home_win_prob
        )
        print(f" ANALYSIS:\n {explanation}")
        print("-" * 53 + "\n  \n")


def main_menu():
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print("=====================================================")
        print("                DIVINELINES V4.0                     ")
        print("=====================================================")
        print(" [1] Sync Database (Fetch Last Night's Games)")
        print(" [2] Launch Predictor Oracle (V4 Optimized)")
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
