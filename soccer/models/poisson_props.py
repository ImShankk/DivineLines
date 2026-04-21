import numpy as np
import pandas as pd
from scipy.stats import poisson

# Required import to link the data engine to the math engine
from soccer.scrapers.ensemble_scraper import build_player_history


def get_implied_probability(american_odds):
    """
    Converts sportsbook American odds into implied probability percentages.
    """
    if american_odds < 0:
        return abs(american_odds) / (abs(american_odds) + 100)
    else:
        return 100 / (american_odds + 100)


def calculate_poisson_probability(lambda_avg, line, bet_type="over"):
    """
    Calculates the Poisson probability of a player prop hitting.
    """
    # poisson.cdf calculates the probability of exactly 'k' or fewer events occurring.
    # np.floor truncates half-point betting lines (e.g., 1.5 becomes 1.0)
    prob_under = poisson.cdf(np.floor(line), lambda_avg)

    if bet_type.lower() == "over":
        return 1 - prob_under
    else:
        return prob_under


def scan_for_value(player_df, prop_col, line, odds, opponent_modifier=1.0):
    """
    Calculates the expected value (EV) by comparing projected probabilities against implied odds.
    """
    # Calculate historical mean, dropping nulls to prevent calculation errors
    raw_avg = player_df[prop_col].dropna().astype(float).mean()

    # Apply external matchup adjustments
    projected_lambda = raw_avg * opponent_modifier

    # Calculate true model probability
    model_prob_over = calculate_poisson_probability(projected_lambda, line, "over")

    # Calculate bookmaker implied probability
    bookie_prob = get_implied_probability(odds)

    # Determine structural edge
    edge = model_prob_over - bookie_prob

    return {
        "projected_avg": round(projected_lambda, 2),
        "model_win_prob": round(model_prob_over * 100, 2),
        "bookie_implied_prob": round(bookie_prob * 100, 2),
        "edge_percentage": round(edge * 100, 2),
        "is_value_bet": edge > 0.03,  # Threshold set to flag edges > 3%
    }


if __name__ == "__main__":
    print("Step 1: Scraping historical player data...")

    # Test array utilizing recent match data
    recent_matches = [
        {
            "fotmob_id": 4217528,
            "sofa_url": "https://www.sofascore.com/manchester-city-real-madrid/hbsEdb",
        }
    ]

    try:
        # Execute the scraper pipeline
        real_data_df = build_player_history(recent_matches)

        target_player = "kevin de bruyne"
        player_stats = real_data_df[real_data_df["name_lower"] == target_player]

        if not player_stats.empty:
            print(f"Step 2: Scanning for value on {target_player.title()}...")

            # Execute Poisson evaluation
            result = scan_for_value(
                player_df=player_stats,
                prop_col="fotmob_sot",
                line=0.5,
                odds=-120,
                opponent_modifier=1.0,
            )

            for key, value in result.items():
                print(f"{key}: {value}")
        else:
            print("Player not found in scraped dataset.")

    except Exception as e:
        print(f"Pipeline failed: {e}")
