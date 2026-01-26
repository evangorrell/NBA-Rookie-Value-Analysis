"""Build training and prediction datasets."""

import pandas as pd
import numpy as np
import os
import pickle
from ..fetch.nba_stats import fetch_player_stats, fetch_advanced_stats, combine_stats
from ..fetch.rookies import fetch_rookie_stats
from ..fetch.salaries import load_rookie_scale_salaries, adjust_salary_for_inflation


def compute_production(df, target_metric='PIE', volume_metric='MIN'):
    """
    Compute production metric as rate * volume.

    Args:
        df: DataFrame with stats
        target_metric: Rate metric (e.g., PIE)
        volume_metric: Volume metric (e.g., MIN for minutes)

    Returns:
        DataFrame with 'production' column added
    """
    # PIE is already a percentage (0-1 scale), so multiply by minutes
    df['production'] = df[target_metric] * df[volume_metric]

    # Handle missing values
    df['production'] = df['production'].fillna(0)

    return df


def add_salary_info(rookies_df, salary_scale_df):
    """
    Add salary information to rookies based on their draft pick.
    Only includes drafted players (picks 1-60).

    Args:
        rookies_df: DataFrame with rookie stats and draft pick
        salary_scale_df: DataFrame with pick -> salary mapping

    Returns:
        DataFrame with salary column (undrafted players excluded)
    """
    # Merge salary info based on draft pick (inner join = only keep drafted)
    result = rookies_df.merge(
        salary_scale_df,
        on='pick',
        how='inner'
    )

    return result


def build_historical_dataset(seasons, current_season, min_games=10):
    """
    Build a dataset of historical rookies with production and salary.
    Salaries are inflation-adjusted to current_season dollars.

    Uses caching to avoid re-fetching data on subsequent runs.

    Args:
        seasons: List of season strings (e.g., ["2019-20", "2020-21"])
        current_season: Current season string for inflation adjustment (e.g., "2025-26")
        min_games: Minimum games played threshold

    Returns:
        DataFrame with columns: player_name, season, pick, salary, production
    """
    # Check for cached historical data
    cache_file = f'outputs/historical_data_{current_season}.pkl'

    if os.path.exists(cache_file):
        print(f"\n✓ Loading cached historical data from {cache_file}")
        with open(cache_file, 'rb') as f:
            dataset = pickle.load(f)
        print(f"✓ Loaded {len(dataset)} rookies across {len(seasons)} seasons from cache")
        return dataset

    print("\nNo cache found. Fetching historical data from NBA API...")

    all_rookies = []

    # Load salary scale (we'll use the same scale for all years for simplicity)
    salary_scale = load_rookie_scale_salaries(current_season)

    for season in seasons:
        print(f"\nProcessing {season}...")

        # Fetch rookie stats
        rookies = fetch_rookie_stats(
            season,
            fetch_player_stats,
            fetch_advanced_stats,
            combine_stats,
            min_games=min_games
        )

        if rookies.empty:
            continue

        # Compute production metric
        rookies = compute_production(rookies)

        # Add salary info
        rookies = add_salary_info(rookies, salary_scale)

        # Adjust salary for inflation to current season dollars
        rookies['salary'] = rookies['salary'].apply(
            lambda x: adjust_salary_for_inflation(x, season, current_season)
        )

        print(f"  Adjusted salaries from {season} to {current_season} dollars (2% annual inflation)")

        # Select key columns
        rookies = rookies[[
            'PLAYER_NAME', 'SEASON', 'pick', 'salary', 'production',
            'GP', 'MIN', 'PIE', 'team_abbrev'
        ]].copy()

        all_rookies.append(rookies)

    # Combine all seasons
    if not all_rookies:
        return pd.DataFrame()

    dataset = pd.concat(all_rookies, ignore_index=True)

    print(f"\n✓ Built historical dataset with {len(dataset)} rookies across {len(seasons)} seasons")

    # Save to cache for future runs
    os.makedirs('outputs', exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"✓ Cached historical data to {cache_file}")

    return dataset


def build_current_dataset(season, min_games=10):
    """
    Build dataset for current season rookies.

    Args:
        season: Season string (e.g., "2025-26")
        min_games: Minimum games played threshold

    Returns:
        DataFrame with current rookies
    """
    print(f"\nProcessing current season: {season}...")

    # Load salary scale
    salary_scale = load_rookie_scale_salaries(season)

    # Fetch rookie stats
    rookies = fetch_rookie_stats(
        season,
        fetch_player_stats,
        fetch_advanced_stats,
        combine_stats,
        min_games=min_games
    )

    if rookies.empty:
        return pd.DataFrame()

    # Compute production metric
    rookies = compute_production(rookies)

    # Add salary info
    rookies = add_salary_info(rookies, salary_scale)

    # Select key columns
    rookies = rookies[[
        'PLAYER_NAME', 'SEASON', 'pick', 'salary', 'production',
        'GP', 'MIN', 'PIE', 'team_abbrev'
    ]].copy()

    print(f"  Built current season dataset with {len(rookies)} rookies")

    return rookies
