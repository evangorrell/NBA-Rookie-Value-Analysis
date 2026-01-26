#!/usr/bin/env python3
"""
NBA Rookie Contract Regression Value Analysis

Analyzes which NBA rookies are providing the most (or least) value
relative to their rookie-scale contracts.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src import config
from src.features.build_dataset import build_historical_dataset, build_current_dataset
from src.model.train import train_model, save_model
from src.model.predict import calculate_residuals, export_residuals
from src.display.residual_chart import create_residual_chart, create_summary_stats
from src.model.diagnostics import create_prediction_accuracy_plot, print_accuracy_examples, print_accuracy_by_salary_range, validate_specific_players


def main():
    """Run the complete analysis pipeline."""
    print("=" * 60)
    print("NBA ROOKIE CONTRACT SURPLUS VALUE ANALYSIS")
    print("=" * 60)

    # Step 1: Build historical dataset
    print("\n[1/6] Building historical training dataset...")
    historical_df = build_historical_dataset(
        seasons=config.HISTORICAL_SEASONS,
        current_season=config.CURRENT_SEASON,
        min_games=config.MIN_GAMES_PLAYED
    )

    if historical_df.empty:
        print("ERROR: No historical data collected. Check API access and season values.")
        return

    print(f"\n  Historical dataset: {len(historical_df)} rookies")
    print(f"  Seasons: {config.HISTORICAL_SEASONS}")

    # Step 2: Train model
    print("\n[2/6] Training regression model...")
    pipeline = train_model(historical_df)

    # Save model
    save_model(pipeline)

    # Step 3: Build current season dataset
    print("\n[3/6] Fetching current season data...")
    current_df = build_current_dataset(
        season=config.CURRENT_SEASON,
        min_games=config.MIN_GAMES_PLAYED
    )

    if current_df.empty:
        print("ERROR: No current season data collected.")
        return

    print(f"\n  Current season: {len(current_df)} rookies")

    # Step 4: Calculate residuals
    print("\n[4/6] Calculating residual values...")
    residuals_df = calculate_residuals(current_df, pipeline)

    # Export residuals
    export_residuals(residuals_df, config.CURRENT_SEASON)

    # Step 5: Create visualization
    print("\n[5/6] Creating visualization...")
    create_residual_chart(
        residuals_df,
        config.CURRENT_SEASON,
        figsize=config.CHART_FIGSIZE,
        surplus_color=config.SURPLUS_COLOR,
        deficit_color=config.DEFICIT_COLOR
    )

    # Step 6: Model accuracy diagnostics
    print("\n[6/6] Running model diagnostics...")
    create_prediction_accuracy_plot(residuals_df, config.CURRENT_SEASON)
    print_accuracy_examples(residuals_df, n=5)
    print_accuracy_by_salary_range(residuals_df)

    # Print summary
    create_summary_stats(residuals_df)

    # Print top performers
    print("\n=== Top 5 Surplus Value Rookies ===")
    top_5 = residuals_df.head(5)
    for _, row in top_5.iterrows():
        print(f"  {row['PLAYER_NAME']:20s} ({row['team_abbrev']:3s}) | "
              f"Pick {row['pick']:2.0f} | Residual: +{row['residual']:.2f}")

    print("\n=== Bottom 5 (Biggest Deficits) ===")
    bottom_5 = residuals_df.tail(5).iloc[::-1]  # Reverse to show worst first
    bottom_5_names = []
    for _, row in bottom_5.iterrows():
        print(f"  {row['PLAYER_NAME']:20s} ({row['team_abbrev']:3s}) | "
              f"Pick {row['pick']:2.0f} | Residual: {row['residual']:.2f}")
        bottom_5_names.append(row['PLAYER_NAME'])

    # Validate bottom performers to explain why they have deficits
    if len(bottom_5_names) > 0:
        print("\n" + "=" * 60)
        print("VALIDATING DEFICIT PLAYERS")
        print("(Understanding why top picks may show negative residuals)")
        print("=" * 60)
        # Validate up to 3 players from bottom to avoid clutter
        validate_specific_players(residuals_df, bottom_5_names[:3], historical_df)

    print("\n" + "=" * 60)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nOutputs:")
    print(f"  - outputs/{config.CURRENT_SEASON}_rookies_residuals.csv")
    print(f"  - outputs/{config.CURRENT_SEASON}_residual_bar_chart.png")
    print(f"  - outputs/{config.CURRENT_SEASON}_accuracy_diagnostic.png")
    print(f"  - outputs/historical_data_{config.CURRENT_SEASON}.pkl (cached)")
    print("  - outputs/model.pkl")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
