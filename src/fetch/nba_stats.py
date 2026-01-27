"""Fetch NBA stats using nba_api."""

import time
import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats
from nba_api.stats.library.parameters import SeasonType


def fetch_player_stats(season, season_type=SeasonType.regular):
    """
    Fetch player stats for a given season.

    Args:
        season: Season string like "2025-26"
        season_type: Regular season or playoffs

    Returns:
        DataFrame with player stats including PIE, minutes, etc.
    """
    print(f"\nProcessing player stats for {season}...")

    try:
        # Fetch traditional stats
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star=season_type,
            per_mode_detailed='Totals',
            timeout=60
        )

        # Get the dataframe
        df = stats.get_data_frames()[0]

        # Add season column
        df['SEASON'] = season

        # Rate limit to NBA API
        time.sleep(0.6)

        return df

    except Exception as e:
        print(f"  Error fetching stats for {season}: {e}")
        return pd.DataFrame()


def fetch_advanced_stats(season, season_type=SeasonType.regular):
    """
    Fetch Player Impact Estimate (PIE).

    Args:
        season: Season string like "2025-26"
        season_type: Regular season or playoffs

    Returns:
        DataFrame with advanced stats
    """
    print(f"Processing advanced stats for {season}...")

    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star=season_type,
            per_mode_detailed='Totals',
            measure_type_detailed_defense='Advanced',
            timeout=60
        )

        df = stats.get_data_frames()[0]
        df['SEASON'] = season

        time.sleep(0.6)

        return df

    except Exception as e:
        print(f"  Error fetching advanced stats for {season}: {e}")
        return pd.DataFrame()


def combine_stats(base_df, advanced_df):
    """
    Combine base and advanced stats.

    Args:
        base_df: Base stats DataFrame
        advanced_df: Advanced stats DataFrame 

    Returns:
        Combined DataFrame
    """
    # Select key columns from advanced stats
    advanced_cols = ['PLAYER_ID', 'SEASON', 'PIE']
    advanced_subset = advanced_df[advanced_cols]

    # Merge on player ID and season
    combined = base_df.merge(
        advanced_subset,
        on=['PLAYER_ID', 'SEASON'],
        how='left'
    )

    return combined
