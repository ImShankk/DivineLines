import json
import pandas as pd
import ScraperFC as sfc
import time
from playwright.sync_api import sync_playwright

sofascore = sfc.Sofascore()


def extract_node(data, target_key):
    """
    Recursively searches JSON and returns the dictionary containing the target key.
    """
    if isinstance(data, dict):
        if target_key in data:
            return data
        for value in data.values():
            res = extract_node(value, target_key)
            if res:
                return res
    elif isinstance(data, list):
        for item in data:
            res = extract_node(item, target_key)
            if res:
                return res
    return None


def parse_fotmob_player_stats(stats_arr, name, team_name, match_id):
    """
    Standardizes stat extraction with defensive type-checking to prevent crashes
    when FotMob alternates between dictionary and raw string values.
    """
    minutes, shots, sot, xg, xa, chances = 0, 0, 0, 0, 0, 0
    passes = "0/0"

    if not isinstance(stats_arr, list):
        return {}  # Failsafe if payload is entirely malformed

    def extract_val(obj):
        """Safely extracts value whether it is nested in a dict or a raw string/int."""
        if isinstance(obj, dict):
            return obj.get("value", 0)
        return obj

    for category in stats_arr:
        if not isinstance(category, dict):
            continue

        cat_stats = category.get("stats", {})

        # Parsing Logic 1: Dictionary format
        if isinstance(cat_stats, dict):
            if "Minutes played" in cat_stats:
                minutes = extract_val(cat_stats["Minutes played"])
            if "Total shots" in cat_stats:
                shots = extract_val(cat_stats["Total shots"])
            if "Shots on target" in cat_stats:
                sot = extract_val(cat_stats["Shots on target"])
            if "Expected goals (xG)" in cat_stats:
                xg = extract_val(cat_stats["Expected goals (xG)"])
            if "Expected assists (xA)" in cat_stats:
                xa = extract_val(cat_stats["Expected assists (xA)"])
            if "Chances created" in cat_stats:
                chances = extract_val(cat_stats["Chances created"])
            if "Accurate passes" in cat_stats:
                passes = extract_val(cat_stats["Accurate passes"])

        # Parsing Logic 2: Array of objects format (Common in MLS)
        elif isinstance(cat_stats, list):
            for stat in cat_stats:
                if isinstance(stat, dict):
                    title = stat.get("title", "")
                    val = stat.get("value", 0)
                    if title == "Minutes played":
                        minutes = val
                    elif title == "Total shots":
                        shots = val
                    elif title == "Shots on target":
                        sot = val
                    elif title == "Expected goals (xG)":
                        xg = val
                    elif title == "Expected assists (xA)":
                        xa = val
                    elif title == "Chances created":
                        chances = val
                    elif title == "Accurate passes":
                        passes = val

    return {
        "name": name,
        "team": team_name,
        "fotmob_match_id": match_id,
        "minutes_played": minutes,
        "shots": shots,
        "sot": sot,
        "xg": xg,
        "xa": xa,
        "chances_created": chances,
        "passes": passes,
    }


def scrape_fotmob_match(match_id):
    """
    Retrieves attacking statistics via Headless Browser by extracting the
    hydration payload directly from the DOM, handling both Opta and non-Opta leagues.
    """
    url = f"https://www.fotmob.com/matches/{match_id}/match"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded")

            script_text = page.locator("#__NEXT_DATA__").inner_text()
            raw_json = json.loads(script_text)
            browser.close()

        player_stats = []

        # Hunt for either the premium Opta stats or the standard lineup stats
        lineup_node = extract_node(raw_json, "lineup")
        player_stats_node = extract_node(raw_json, "playerStats")

        if not lineup_node and not player_stats_node:
            print(f"Error: No statistical data found for Match {match_id}.")
            return pd.DataFrame()

        # Extract team names for identification
        team_names = {"home": "Unknown", "away": "Unknown"}
        match_facts_node = extract_node(raw_json, "matchFacts")
        if match_facts_node and "matchFacts" in match_facts_node:
            info = match_facts_node["matchFacts"].get("info", {})
            team_names["home"] = info.get("homeTeam", {}).get("name", "Unknown")
            team_names["away"] = info.get("awayTeam", {}).get("name", "Unknown")

        # Parsing Logic 1: Opta Games (Premier League, Champions League, etc.)
        if player_stats_node and "playerStats" in player_stats_node:
            for team in ["home", "away"]:
                players = player_stats_node["playerStats"].get(team, [])
                for player in players:
                    p_data = parse_fotmob_player_stats(
                        player.get("stats", []),
                        player.get("name", "Unknown"),
                        team_names[team],
                        match_id,
                    )
                    player_stats.append(p_data)

        # Parsing Logic 2: Non-Opta Games (MLS, Internationals, Lower Leagues)
        elif lineup_node and "lineup" in lineup_node:
            teams = lineup_node["lineup"].get("lineup", [])
            for team_data in teams:
                team_name = team_data.get("teamName", "Unknown")

                # Extract starters
                for row in team_data.get("players", []):
                    for player in row:
                        name = player.get("name", {}).get("fullName", "Unknown")
                        p_data = parse_fotmob_player_stats(
                            player.get("stats", []), name, team_name, match_id
                        )
                        player_stats.append(p_data)

                # Extract bench players
                for player in team_data.get("bench", []):
                    name = player.get("name", {}).get("fullName", "Unknown")
                    p_data = parse_fotmob_player_stats(
                        player.get("stats", []), name, team_name, match_id
                    )
                    player_stats.append(p_data)

        return pd.DataFrame(player_stats)
    except Exception as e:
        print(f"Failed to scrape FotMob match {match_id}: {e}")
        return pd.DataFrame()


def scrape_sofascore_match(match_url):
    try:
        df = sofascore.get_player_stats(match_url)

        prop_cols = [
            "player_name",
            "position",
            "tackles",
            "interceptions",
            "dribble_attempts",
            "succ_dribbles",
            "duels_won",
            "fouls",
        ]

        available_cols = [col for col in prop_cols if col in df.columns]
        df = df[available_cols].copy()
        df["sofascore_url"] = match_url

        return df
    except Exception as e:
        print(f"Failed to scrape Sofascore match {match_url}: {e}")
        return pd.DataFrame()


def build_player_history(match_list):
    all_merged_games = []

    for match in match_list:
        print(f"Scraping match data for FotMob ID: {match['fotmob_id']}...")

        df_fotmob = scrape_fotmob_match(match["fotmob_id"])

        if df_fotmob.empty:
            print("Missing FotMob data for this match. Skipping.")
            continue

        if match["sofa_url"] == "placeholder":
            all_merged_games.append(df_fotmob)
            time.sleep(2)
            continue

        df_sofa = scrape_sofascore_match(match["sofa_url"])

        if df_sofa.empty:
            all_merged_games.append(df_fotmob)
            time.sleep(2)
            continue

        df_fotmob["join_name"] = df_fotmob["name"].str.lower().str.strip()
        df_sofa["join_name"] = df_sofa["player_name"].str.lower().str.strip()

        merged_df = pd.merge(df_fotmob, df_sofa, on="join_name", how="inner")
        merged_df = merged_df.drop(columns=["join_name", "player_name"])
        all_merged_games.append(merged_df)

        time.sleep(2)

    if all_merged_games:
        return pd.concat(all_merged_games, ignore_index=True)
    else:
        return pd.DataFrame()
