"""
NBA Rookie Contract Regression Value Analysis

Analyzes which NBA rookies are providing the most (or least) value
relative to their rookie-scale contracts.
"""

import sys
from src import config
from src.features.build_dataset import build_historical_dataset, build_current_dataset
from src.model.train import train_model, save_model
from src.model.predict import calculate_residuals, export_residuals
from src.display.residual_chart import create_residual_chart, create_summary_stats
from src.model.diagnostics import create_prediction_accuracy_plot, validate_specific_players


def main():
    """Run the complete analysis pipeline."""
    print("=" * 60)
    print("NBA ROOKIE CONTRACT VALUE ANALYSIS")
    print("=" * 60)

    # Step 1: Build historical dataset
    print("\nBuilding historical training dataset...")
    historical_df = build_historical_dataset(
        seasons=config.HISTORICAL_SEASONS,
        current_season=config.CURRENT_SEASON,
        min_games=config.MIN_GAMES_PLAYED
    )

    if historical_df.empty:
        print("ERROR: No historical data collected. Check API access and season values.")
        return

    print(f"  Historical dataset: {len(historical_df)} rookies")
    print(f"  Seasons: {config.HISTORICAL_SEASONS}")

    # Step 2: Train model
    pipeline = train_model(historical_df)

    # Save model
    save_model(pipeline)

    # Step 3: Build current season dataset
    current_df = build_current_dataset(
        season=config.CURRENT_SEASON,
        min_games=config.MIN_GAMES_PLAYED
    )

    if current_df.empty:
        print("ERROR: No current season data collected.")
        return

    # Step 4: Calculate residuals
    residuals_df = calculate_residuals(current_df, pipeline)

    # Export residuals
    export_residuals(residuals_df, config.CURRENT_SEASON)

    # Step 5: Create visualization
    create_residual_chart(
        residuals_df,
        config.CURRENT_SEASON,
        figsize=config.CHART_FIGSIZE,
        surplus_color=config.SURPLUS_COLOR,
        deficit_color=config.DEFICIT_COLOR
    )

    # Step 6: Model accuracy diagnostics
    print("\nRunning accuracy diagnostics...")
    create_prediction_accuracy_plot(residuals_df, config.CURRENT_SEASON)

    # Print summary
    create_summary_stats(residuals_df)

    # Print top performers
    print("\n  Top 5 Surplus Value Rookies")
    top_5 = residuals_df.head(5)
    for _, row in top_5.iterrows():
        print(f"  {row['PLAYER_NAME']:20s} ({row['team_abbrev']:3s}) | "
              f"Pick {row['pick']:2.0f} | Residual: +{row['residual']:.2f}")

    print("\n  Bottom 5 (Biggest Deficits) Rookies")
    bottom_5 = residuals_df.tail(5).iloc[::-1]  # Reverse to show worst first
    bottom_5_names = []
    for _, row in bottom_5.iterrows():
        print(f"  {row['PLAYER_NAME']:20s} ({row['team_abbrev']:3s}) | "
              f"Pick {row['pick']:2.0f} | Residual: {row['residual']:.2f}")
        bottom_5_names.append(row['PLAYER_NAME'])

    # Prompt user for custom player breakdowns
    print("\n=== Player Breakdown ===")

    while True:
        user_input = input("\nWould you like a detailed breakdown for any specific player(s)?\n"
                           "Enter player name(s) separated by commas, or press Enter to skip: ").strip()

        # If Enter, skip
        if not user_input:
            break

        # Parse comma-separated names
        player_names = [name.strip() for name in user_input.split(',') if name.strip()]

        # If no valid names after parsing
        if not player_names: 
            print(f"\n  No valid player names provided.")
            print(f"  Please try again or press Enter to skip.")
            continue

        # Check if any players match
        found_any = False
        for player_name in player_names:
            matches = residuals_df[residuals_df['PLAYER_NAME'].str.contains(player_name, case=False, na=False)]
            if not matches.empty:
                found_any = True
                break

        if found_any:
            # At least one player found, proceed with breakdown
            validate_specific_players(residuals_df, player_names, historical_df)
            break
        else:
            # No matches found, reprompt
            print(f"\n  No players found matching: {', '.join(player_names)}")
            print(f"  Please try again with different name(s) or press Enter to skip.")

    print("\nANALYSIS COMPLETE")
    first_season = config.HISTORICAL_SEASONS[0] if config.HISTORICAL_SEASONS else "none"
    last_season = config.HISTORICAL_SEASONS[-1] if config.HISTORICAL_SEASONS else "none"

    print("\nOutputs:")
    print(f"  - outputs/{config.CURRENT_SEASON}_rookies_residuals.csv")
    print(f"  - outputs/{config.CURRENT_SEASON}_residual_bar_chart.png")
    print(f"  - outputs/{config.CURRENT_SEASON}_accuracy_diagnostic.png")
    print(f"  - outputs/historical_data_{first_season}_to_{last_season}_for_{config.CURRENT_SEASON}.pkl (cached)")
    print("  - outputs/model.pkl")

    # Keep matplotlib window open
    input("\nPress Enter to exit and close chart...")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
