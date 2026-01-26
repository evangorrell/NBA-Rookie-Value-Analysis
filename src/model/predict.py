"""Calculate predictions and residuals for rookies based on contract."""

import pandas as pd
import numpy as np


def calculate_residuals(current_df, pipeline):
    """
    Calculate residual values for current rookies.

    Residual = Actual Production - Expected Production

    Args:
        current_df: DataFrame with current season rookies
        pipeline: Trained sklearn pipeline

    Returns:
        DataFrame with residuals added
    """
    print("\nCalculating Residuals...")

    # Make predictions
    X = current_df[['salary']].values
    expected_production = pipeline.predict(X)

    # Calculate residuals
    current_df = current_df.copy()
    current_df['expected_production'] = expected_production
    current_df['residual'] = current_df['production'] - current_df['expected_production']

    # Sort by residual (highest surplus first)
    current_df = current_df.sort_values('residual', ascending=False)

    print(f"  Calculated residuals for {len(current_df)} rookies")
    print(f"  Top surplus: {current_df.iloc[0]['PLAYER_NAME']} (+{current_df.iloc[0]['residual']:.2f})")
    print(f"  Biggest deficit: {current_df.iloc[-1]['PLAYER_NAME']} ({current_df.iloc[-1]['residual']:.2f})")

    return current_df


def export_residuals(residuals_df, current_season, filepath=None):
    """
    Export residuals to CSV.

    Args:
        residuals_df: DataFrame with residuals
        current_season: Current season string (e.g., "2025-26")
        filepath: Output filepath (auto-generated if None)
    """
    if filepath is None:
        filepath = f'outputs/{current_season}_rookies_residuals.csv'

    # Select key columns for export
    export_df = residuals_df[[
        'PLAYER_NAME', 'team_abbrev', 'pick', 'salary',
        'GP', 'MIN', 'PIE', 'production',
        'expected_production', 'residual'
    ]].copy()

    # Rename columns
    export_df.columns = [
        'Player', 'Team', 'Pick', 'Salary',
        'Games', 'Minutes', 'PIE', 'Production',
        'Expected', 'Residual'
    ]

    # Save to CSV
    export_df.to_csv(filepath, index=False)
    print(f"✓ Residuals exported to {filepath}")

    return export_df
