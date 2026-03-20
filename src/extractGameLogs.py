import os
import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog


def extract_multiple_seasons(seasons: list, output_dir: str = "../data/raw") -> None:
    """
    Extracts game logs for multiple NBA seasons and saves them as separate CSVs.

    Args:
        seasons (list): A list of season strings (e.g., ["2023-24", "2024-25"]).
        output_dir (str): The folder to save the CSV files.
    """
    print(
        f"DivineLines: Initializing historical data extraction for {len(seasons)} seasons..."
    )
    os.makedirs(output_dir, exist_ok=True)

    for season in seasons:
        print(f"-> Pulling {season} season data...")
        try:
            # Ping the API for this specific season
            game_log = leaguegamelog.LeagueGameLog(
                season=season, player_or_team_abbreviation="T"
            )
            games_df = game_log.get_data_frames()[0]

            # Create a clean file name by replacing "-" with "_"
            safe_season_name = season.replace("-", "_")
            file_path = os.path.join(output_dir, f"nba_games_{safe_season_name}.csv")

            # Save to CSV
            games_df.to_csv(file_path, index=False)
            print(f"   [SUCCESS] Saved {len(games_df)} games to {file_path}")

            # CRITICAL: Sleep for 3 seconds between API calls to avoid rate limits
            time.sleep(3)

        except Exception as e:
            print(f"   [FAILED] Could not pull {season}: {e}")


if __name__ == "__main__":
    # Define our Train and Test buckets here
    target_seasons = [
        "2021-22",  # Training Data
        "2022-23",  # Training Data
        "2023-24",  # Training Data
        "2024-25",  # Backtesting Data
        "2025-26",  # Live Season Data
    ]

    extract_multiple_seasons(target_seasons)
