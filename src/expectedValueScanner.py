import sqlite3
import pandas as pd
import xgboost as xgb
import os


def get_latest_team_stats(db_path: str, team_abbr: str):
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM game_logs WHERE TEAM_ABBREVIATION = '{team_abbr}' ORDER BY GAME_DATE ASC"
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        return None

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["WIN_BIN"] = df["WL"].apply(lambda x: 1 if x == "W" else 0)

    last_10 = df.tail(10)
    stats = {}

    # 1. Core Scoring & Efficiency
    stats["ROLL_10_PTS"] = last_10["PTS"].mean()
    stats["ROLL_10_PLUS_MINUS"] = last_10["PLUS_MINUS"].mean()
    stats["ROLL_10_FG_PCT"] = last_10["FG_PCT"].mean()
    stats["ROLL_10_FG3_PCT"] = last_10["FG3_PCT"].mean()
    stats["ROLL_10_FT_PCT"] = last_10["FT_PCT"].mean()

    # 2. Rebounding & Possession
    stats["ROLL_10_REB"] = last_10["REB"].mean()
    stats["ROLL_10_OREB"] = last_10["OREB"].mean()
    stats["ROLL_10_AST"] = last_10["AST"].mean()
    stats["ROLL_10_TOV"] = last_10["TOV"].mean()

    # 3. Defense
    stats["ROLL_10_STL"] = last_10["STL"].mean()
    stats["ROLL_10_BLK"] = last_10["BLK"].mean()
    stats["ROLL_10_PF"] = last_10["PF"].mean()

    # 4. Momentum & Rest
    stats["ROLL_10_WIN_PCT"] = last_10["WIN_BIN"].mean()
    last_game_date = df.iloc[-1]["GAME_DATE"]
    days_since_last_game = (pd.Timestamp.today() - last_game_date).days
    stats["DAYS_REST"] = min(max(days_since_last_game, 0), 4)

    return stats


def get_h2h_win_pct(db_path: str, home_team: str, away_team: str):
    """Calculates the Home Team's win percentage against the Away Team this season."""
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
        return 0.50  # Assume 50% if no head-to-head data exists this season

    wins = len(df[df["WL"] == "W"])
    total_games = len(df)
    return wins / total_games


def convert_moneyline_to_prob(moneyline: int) -> float:
    if moneyline < 0:
        return (-moneyline) / (-moneyline + 100) * 100
    else:
        return 100 / (moneyline + 100) * 100


def run_ev_scanner():
    db_path = os.path.join("..", "data", "processed", "nba_data.db")

    model_path = os.path.join("..", "data", "processed", "divinelines_v3.json")

    print("Loading DivineLines EV Engine...")
    model = xgb.Booster()

    if not os.path.exists(model_path):
        print(
            f"[!] Could not find {model_path}. Please run trainModel.py first to train and save the model."
        )
        return

    model.load_model(model_path)

    os.system("cls" if os.name == "nt" else "clear")
    print("=====================================================")
    print("             EXPECTED VALUE SCANNER             ")
    print("=====================================================")

    while True:
        try:
            print("\n--- New Matchup ---")
            away_team = input("Away Team (e.g., DAL) or 'exit': ").strip().upper()
            if away_team in ["EXIT", "QUIT"]:
                break

            home_team = input("Home Team (e.g., BOS): ").strip().upper()
            if home_team in ["EXIT", "QUIT"]:
                break

            if home_team == away_team:
                print(
                    f"\n[!] ERROR: {home_team} cannot play themselves. Please enter two different teams."
                )
                continue

            away_odds = int(input(f"{away_team} Moneyline (e.g., +130): "))
            home_odds = int(input(f"{home_team} Moneyline (e.g., -150): "))

        except ValueError:
            print("[!] Please enter a valid number for the odds.")
            continue

        home_prob = convert_moneyline_to_prob(home_odds)
        away_prob = convert_moneyline_to_prob(away_odds)

        home_stats = get_latest_team_stats(db_path, home_team)
        away_stats = get_latest_team_stats(db_path, away_team)
        h2h_home_win_pct = get_h2h_win_pct(db_path, home_team, away_team)

        if not home_stats or not away_stats:
            print(f"[!] Data missing for {away_team} or {home_team}.")
            continue

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
            "H2H_WIN_PCT": h2h_home_win_pct,
        }

        # Convert to DataFrame ensuring exact column order
        matchup_df = pd.DataFrame([diff_features])
        dmatrix = xgb.DMatrix(matchup_df)

        home_ai_prob = model.predict(dmatrix)[0] * 100
        away_ai_prob = 100 - home_ai_prob

        home_edge = home_ai_prob - home_prob
        away_edge = away_ai_prob - away_prob

        print("\n" + "=" * 53)
        print(f"[{away_team} @ {home_team} - VALUE ANALYSIS]")
        print("-" * 53)
        print(f"{home_team} (HOME):")
        print(f"Probability: {home_prob:.1f}% (Odds: {home_odds})")
        print(f"AI Probability:    {home_ai_prob:.1f}%")
        if home_edge > 0:
            print(f"EDGE: +{home_edge:.1f}% VALUE DETECTED")
        else:
            print(f"EDGE: {home_edge:.1f}% (Bad Bet)")

        print(f"\n{away_team} (AWAY):")
        print(f"Probability: {away_prob:.1f}% (Odds: {away_odds})")
        print(f"AI Probability:    {away_ai_prob:.1f}%")
        if away_edge > 0:
            print(f"EDGE: +{away_edge:.1f}% VALUE DETECTED")
        else:
            print(f"EDGE: {away_edge:.1f}% (Bad Bet)")
        print("=" * 53)


if __name__ == "__main__":
    run_ev_scanner()
