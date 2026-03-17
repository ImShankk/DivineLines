import os
import pandas as pd
from nba_api.stats.static import teams


def extract_team_data(output_path: str = "../data/raw/nba_teams.csv") -> None:
    """
    Extracts active NBA team data and saves it as a CSV file.

    Args:
        output_path (str): The full path where the CSV file will be saved.
    """
    print("DivineLines: Extracting NBA team data...")

    # 1. Get the list of NBA teams
    nba_teams = teams.get_teams()

    # 2. Convert the list of teams to a DataFrame
    teams_df = pd.DataFrame(nba_teams)

    # 3. Create the directories if they don't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 4. Save the DataFrame directly to the output_path
    teams_df.to_csv(output_path, index=False)

    print(f"Success! {len(teams_df)} NBA teams saved to {output_path}")
    print("\nSample Data:")
    print(teams_df[["id", "full_name", "abbreviation"]].head())


if __name__ == "__main__":
    extract_team_data()
