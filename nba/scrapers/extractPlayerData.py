import os
import pandas as pd
from nba_api.stats.static import players


def extract_player_data(output_path: str = "../data/raw/nba_players.csv") -> None:
    """
    Extracts active NBA player data and saves it as a CSV file.

    Args:
        output_path (str): The full path where the CSV file will be saved.
    """
    print("DivineLines: Extracting NBA player data...")

    # 1. Get the list of NBA players
    nba_players = players.get_players()

    active_players = [player for player in nba_players if player["is_active"]]

    # 2. Convert the list of players to a DataFrame
    players_df = pd.DataFrame(active_players)

    # 3. Create the directories if they don't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    file_path = os.path.join(os.path.dirname(output_path), "nba_players.csv")

    # 4. Save the DataFrame directly to the output_path
    players_df.to_csv(output_path, index=False)

    print(f"Success! {len(players_df)} NBA players saved to {output_path}")


if __name__ == "__main__":
    extract_player_data()
