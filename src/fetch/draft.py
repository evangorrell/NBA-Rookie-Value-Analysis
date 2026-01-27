"""Fetch NBA draft data."""

import pandas as pd
from nba_api.stats.endpoints import drafthistory
import time


def fetch_draft_class(season):
    """
    Fetch draft data for a given season.

    Args:
        season: Season string like "2025-26"

    Returns:
        DataFrame with draft picks including: player_name, pick, team
    """
    # Extract the draft year from season (e.g., "2025-26" -> 2025)
    draft_year = int(season.split('-')[0])

    print(f"Fetching data for {draft_year} draft...")

    try:
        # Fetch all draft history
        draft = drafthistory.DraftHistory(
            season_year_nullable=str(draft_year),
            timeout=60
        )

        df = draft.get_data_frames()[0]

        # Clean up the data
        df = df.rename(columns={
            'PERSON_ID': 'player_id',
            'PLAYER_NAME': 'player_name',
            'OVERALL_PICK': 'pick',
            'TEAM_NAME': 'team',
            'TEAM_ABBREVIATION': 'team_abbrev'
        })

        # Select relevant columns
        df = df[['player_id', 'player_name', 'pick', 'team', 'team_abbrev']].copy()
        df['draft_year'] = draft_year

        print(f"  Retrieved {len(df)} draft picks from {draft_year}")

        time.sleep(0.6)

        return df

    except Exception as e:
        print(f"  Error fetching draft data for {draft_year}: {e}")
        return pd.DataFrame()


def get_rookie_player_stats_draft(stats_df, draft_df):
    """
    Merge player stats with draft data to identify and filter drafted rookies.

    Args:
        stats_df: DataFrame with all player stats
        draft_df: DataFrame with draft picks

    Returns:
        DataFrame with only rookies, including their draft pick and salary info
    """
    # Merge stats with draft data
    rookies = stats_df.merge(
        draft_df,
        left_on='PLAYER_ID',
        right_on='player_id',
        how='inner'  # Only keep players who were drafted
    )

    print(f"  Matched {len(rookies)} rookies from draft to stats")

    return rookies
