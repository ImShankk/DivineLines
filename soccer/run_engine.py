import sys
import pandas as pd

from soccer.scrapers.player_profile import get_player_overview
from soccer.scrapers.ensemble_scraper import build_player_history
from soccer.models.poisson_props import scan_for_value


def display_menu():
    """
    Standardizes the available prop metrics for the terminal UI.
    """
    print("\n--- Available Prop Markets ---")
    print("1. Shots on Target (fotmob_sot)")
    print("2. Total Shots (shots)")
    print("3. Accurate Passes (passes)")
    print("4. Chances Created (chances_created)")
    print("5. Tackles (tackles)")
    print("------------------------------")


def main():
    """
    Main execution loop prompting for dynamic user input to scan live betting lines.
    """
    print("\n=======================================")
    print("  DIVINELINES: SOCCER QUANT ENGINE")
    print("=======================================\n")

    # 1. Player Input
    # Note: FotMob IDs can be found by searching a player on fotmob.com
    # and looking at the number in the URL (e.g., /players/737066/erling-haaland)
    try:
        player_id = int(input("Enter FotMob Player ID: ").strip())
    except ValueError:
        print("Error: Player ID must be a number.")
        sys.exit()

    # 2. Fetch Match History
    print(f"\n[1/3] Locating recent matches for ID {player_id}...")
    profile = get_player_overview(player_id)

    if not profile or not profile.get("recent_matches"):
        print("Error: Could not retrieve player profile or recent matches.")
        sys.exit()

    print(f"Target Acquired: {profile['player_name']} ({profile['team']})")

    # 3. Scrape Deep Stats
    print("\n[2/3] Scraping Opta and Sofascore databases...")
    # Filter for games where the player actually logged minutes
    active_matches = [m for m in profile["recent_matches"] if m["minutes"] > 0]

    # The scraper requires dicts with fotmob_id and sofa_url.
    # Since we are automating, we use a placeholder for SofaScore URLs for now
    # unless we build a dedicated Sofascore searcher later.
    match_payload = [
        {"fotmob_id": m["match_id"], "sofa_url": "placeholder"} for m in active_matches
    ]

    historical_df = build_player_history(match_payload)

    if historical_df.empty:
        print("Error: Failed to build historical dataset.")
        sys.exit()

    # 4. Betting Inputs
    display_menu()
    prop_col = input("Enter the exact prop market string (e.g., fotmob_sot): ").strip()

    if prop_col not in historical_df.columns:
        print(f"Error: Metric '{prop_col}' not found in scraped data.")
        sys.exit()

    try:
        line = float(input("Enter the Sportsbook Line (e.g., 0.5): ").strip())
        odds = int(input("Enter the American Odds (e.g., -110): ").strip())
        modifier = float(
            input("Enter Opponent Modifier (Default 1.0 for neutral): ").strip()
            or "1.0"
        )
    except ValueError:
        print("Error: Invalid numerical input for betting lines.")
        sys.exit()

    # 5. Execute Mathematical Model
    print("\n[3/3] Running Poisson Distribution Model...")

    # Extract the player's last name for safer, fuzzy matching
    target_last_name = profile["player_name"].split()[-1].lower().strip()

    # Filter the dataframe for any name containing the target last name
    player_data = historical_df[
        historical_df["name"].str.lower().str.contains(target_last_name, na=False)
    ]

    if player_data.empty:
        print(
            f"Error: Could not isolate {profile['player_name']} in the scraped game logs."
        )
        sys.exit()

    result = scan_for_value(
        player_df=player_data,
        prop_col=prop_col,
        line=line,
        odds=odds,
        opponent_modifier=modifier,
    )

    # 6. Output Terminal Report
    print("\n=======================================")
    print("        DIVINELINES EDGE REPORT        ")
    print("=======================================")
    print(f"Player:     {profile['player_name']}")
    print(f"Market:     Over {line} {prop_col}")
    print(
        f"Base Avg:   {result['raw_historical_avg']} (Last {len(active_matches)} matches)"
    )
    print(f"Proj Avg:   {result['projected_avg']} (After modifier)")
    print("---------------------------------------")
    print(f"Model Prob: {result['model_win_prob']}%")
    print(f"Impl Odds:  {result['bookie_implied_prob']}%")

    edge_str = f"{result['edge_percentage']}%"
    if result["is_value_bet"]:
        print(f"EDGE FOUND: +{edge_str} [VALUE BET DETECTED]")
    else:
        print(f"NO EDGE:    {edge_str} [AVOID BET]")
    print("=======================================\n")


if __name__ == "__main__":
    main()
