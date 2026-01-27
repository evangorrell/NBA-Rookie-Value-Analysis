"""Identify and fetch rookie player data."""

import pandas as pd
from .draft import fetch_draft_class, get_rookie_player_stats_draft


def fetch_rookie_stats(season, base_fetcher, advanced_fetcher, combiner, min_games=10):
    """
    Fetch regular season stats for rookies in a given season using draft data.

    NOTE: Only regular season stats are used. Playoffs are excluded because not all rookies make the playoffs

    Args:
        season: Season string (e.g., "2024-25" for 2024 draft class)
        base_fetcher: Function to fetch base stats
        advanced_fetcher: Function to fetch advanced stats 
        combiner: Function to combine stats
        min_games: Minimum games played filter

    Returns:
        DataFrame with rookie regular season stats, draft pick, and team info
    """
    # Fetch all player stats for the season
    base_stats = base_fetcher(season) # Player info, games, minutes, team
    advanced_stats = advanced_fetcher(season) # Player impact estimate (PIE)

    if base_stats.empty or advanced_stats.empty:
        return pd.DataFrame()

    # Combine base and advanced stats
    combined = combiner(base_stats, advanced_stats)

    # Fetch draft data for this season's draft class
    draft_data = fetch_draft_class(season)

    if draft_data.empty:
        print(f"  Warning: No draft data found for {season}")
        return pd.DataFrame()

    # Filter stats to only drafted rookies (ignore undrafted rookies, G-league call-ups, etc.)
    rookies = get_rookie_player_stats_draft(combined, draft_data)

    # Filter by minimum games played
    rookies = rookies[rookies['GP'] >= min_games].copy()

    print(f"  {len(rookies)} rookies with {min_games}+ games")

    return rookies
