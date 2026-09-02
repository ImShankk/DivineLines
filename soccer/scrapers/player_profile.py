import json
from playwright.sync_api import sync_playwright


def extract_node(data, target_key):
    """
    Recursively searches a JSON dictionary and returns the object containing the target key.
    This bypasses any hidden wrappers or dynamic Next.js structures.
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


def get_player_overview(player_id):
    """
    Scrapes the frontend player page and extracts the Next.js hydration payload.
    """
    # The frontend URL automatically redirects to the correct player slug
    url = f"https://www.fotmob.com/players/{player_id}/player"

    try:
        print(f"Fetching profile via frontend for Player ID: {player_id}...")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded")

            # Extract the raw JSON embedded directly in the HTML source
            script_text = page.locator("#__NEXT_DATA__").inner_text()
            raw_json = json.loads(script_text)
            browser.close()

        # Dynamically hunt for the player data payload
        player_data = extract_node(raw_json, "recentMatches")

        if not player_data:
            print("Error: 'recentMatches' payload missing from frontend cache.")
            return None

        name = player_data.get("name", "Unknown")
        team = player_data.get("primaryTeam", {}).get("teamName", "Unknown")

        recent_matches = []
        for match in player_data.get("recentMatches", [])[:5]:
            recent_matches.append(
                {
                    "match_id": match.get("id"),
                    "match_date": match.get("matchDate"),
                    "opponent": match.get("opponentTeamName"),
                    "minutes": match.get("minutesPlayed", 0),
                }
            )

        season_stats = {}
        stat_seasons = player_data.get("statSeasons", [])
        if stat_seasons:
            tournaments = stat_seasons[0].get("tournaments", [])
            if tournaments:
                season_stats = tournaments[0]

        return {
            "player_name": name,
            "team": team,
            "recent_matches": recent_matches,
            "raw_season_data": season_stats,
        }

    except Exception as e:
        print(f"Error extracting player {player_id}: {e}")
        return None
