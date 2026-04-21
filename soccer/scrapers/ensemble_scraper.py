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


def scrape_fotmob_match(match_id):
    """
    Retrieves Opta attacking statistics via Headless Browser.
    Bounces off the main frontend to clear Cloudflare before hitting the API.
    """
    api_url = f"https://www.fotmob.com/api/matchDetails?matchId={match_id}"

    try:
        with sync_playwright() as p:
            # Adding standard user agent to avoid basic bot detection
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = context.new_page()

            # 1. Navigate to the homepage to establish a trusted Cloudflare session
            page.goto("https://www.fotmob.com", wait_until="domcontentloaded")

            # 2. Navigate to the raw API endpoint using the cleared session
            page.goto(api_url, wait_until="domcontentloaded")

            # Extract JSON directly from the browser body
            raw_json = page.locator("body").inner_text()
            data = json.loads(raw_json)
            browser.close()

        # Validate that the API returned the expected stats payload
        if "content" not in data or "playerStats" not in data.get("content", {}):
            print(f"Error: 'playerStats' missing from Match {match_id} payload.")
            return pd.DataFrame()

        player_stats = []

        for team in ["home", "away"]:
            players = data["content"]["playerStats"].get(team, [])

            team_name = "Unknown"
            if (
                "matchFacts" in data.get("content", {})
                and "info" in data["content"]["matchFacts"]
            ):
                team_name = (
                    data["content"]["matchFacts"]["info"]
                    .get(team + "Team", {})
                    .get("name", "Unknown")
                )

            for player in players:
                stats_arr = player.get("stats", [])

                minutes = 0
                shots = 0
                sot = 0
                xg = 0
                xa = 0
                chances = 0
                passes = "0/0"

                for category in stats_arr:
                    cat_stats = category.get("stats", {})
                    if "Minutes played" in cat_stats:
                        minutes = cat_stats["Minutes played"].get("value", 0)
                    if "Total shots" in cat_stats:
                        shots = cat_stats["Total shots"].get("value", 0)
                    if "Shots on target" in cat_stats:
                        sot = cat_stats["Shots on target"].get("value", 0)
                    if "Expected goals (xG)" in cat_stats:
                        xg = cat_stats["Expected goals (xG)"].get("value", 0)
                    if "Expected assists (xA)" in cat_stats:
                        xa = cat_stats["Expected assists (xA)"].get("value", 0)
                    if "Chances created" in cat_stats:
                        chances = cat_stats["Chances created"].get("value", 0)
                    if "Accurate passes" in cat_stats:
                        passes = cat_stats["Accurate passes"].get("value", "0/0")

                p_data = {
                    "name": player.get("name", "Unknown"),
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
