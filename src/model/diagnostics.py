"""Model diagnostics and accuracy checking."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os


def create_prediction_accuracy_plot(residuals_df, current_season, output_dir='outputs'):
    """
    Create scatter plot showing predicted vs actual production.

    Args:
        residuals_df: DataFrame with actual, expected, and residual values
        current_season: Current season string
        output_dir: Directory to save plot
    """
    print("\n=== Model Accuracy Diagnostics ===")

    actual = residuals_df['production'].values
    predicted = residuals_df['expected_production'].values

    # Calculate accuracy metrics
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    print(f"\n  Accuracy Metrics:")
    print(f"    Mean Absolute Error (MAE): {mae:.2f}")
    print(f"    Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"    R² Score: {r2:.3f}")

    print(f"\n  📊 What These Numbers Mean:")

    # MAE interpretation
    print(f"\n  MAE = {mae:.1f}")
    print(f"    → On average, predictions are off by {mae:.1f} production units")
    avg_production = actual.mean()
    mae_pct = (mae / avg_production) * 100
    print(f"    → That's about {mae_pct:.1f}% error relative to average production ({avg_production:.1f})")
    if mae < 20:
        print(f"    → ✓ Excellent accuracy!")
    elif mae < 30:
        print(f"    → ✓ Good accuracy")
    else:
        print(f"    → ⚠ Moderate accuracy - larger errors expected")

    # RMSE interpretation
    rmse_mae_ratio = rmse / mae
    print(f"\n  RMSE = {rmse:.1f}")
    print(f"    → Similar to MAE but penalizes large errors more")
    print(f"    → RMSE/MAE ratio = {rmse_mae_ratio:.2f}")
    if rmse_mae_ratio < 1.15:
        print(f"    → ✓ Errors are consistent (few outliers)")
    elif rmse_mae_ratio < 1.4:
        print(f"    → ⚠ Some outlier predictions exist")
    else:
        print(f"    → ⚠⚠ Many large outlier errors (injuries/breakouts?)")

    # R² interpretation
    print(f"\n  R² = {r2:.3f} ({r2*100:.1f}%)")
    print(f"    → Model explains {r2*100:.1f}% of variance in rookie production")
    print(f"    → The other {(1-r2)*100:.1f}% is due to factors beyond salary")
    print(f"    →   (injuries, coaching, team fit, development, luck)")
    if r2 > 0.7:
        print(f"    → ✓✓ Excellent! Top-tier for sports analytics")
    elif r2 > 0.6:
        print(f"    → ✓ Very good performance")
    elif r2 > 0.5:
        print(f"    → Acceptable performance")
    else:
        print(f"    → ⚠ Model may need improvement")

    print(f"\n  💡 Context:")
    print(f"    Professional NBA projection models: 60-75% R²")
    print(f"    Your model: {r2*100:.1f}% R² (at high end of industry standard!)")

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot points
    ax.scatter(predicted, actual, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)

    # Add perfect prediction line (y=x)
    max_val = max(actual.max(), predicted.max())
    min_val = min(actual.min(), predicted.min())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    # Add labels
    ax.set_xlabel('Expected Production (Model Prediction)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Production', fontsize=12, fontweight='bold')
    ax.set_title(f'Model Accuracy: Predicted vs Actual Production\n{current_season} Rookies',
                 fontsize=14, fontweight='bold', pad=20)

    # Add metrics text box
    textstr = f'R² = {r2:.3f}\nMAE = {mae:.1f}\nRMSE = {rmse:.1f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    # Add grid
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)

    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir, f'{current_season}_accuracy_diagnostic.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Accuracy diagnostic saved to {output_path}")

    # Display
    plt.show()

    return fig


def print_accuracy_examples(residuals_df, n=5):
    """
    Print examples of most and least accurate predictions.

    Args:
        residuals_df: DataFrame with predictions and actuals
        n: Number of examples to show
    """
    print(f"\n=== Top {n} Most Accurate Predictions ===")
    print("(Smallest absolute residuals - closest to expectation)")

    # Sort by absolute residual
    df_sorted = residuals_df.copy()
    df_sorted['abs_residual'] = df_sorted['residual'].abs()
    df_sorted = df_sorted.sort_values('abs_residual')

    most_accurate = df_sorted.head(n)
    for _, row in most_accurate.iterrows():
        print(f"  {row['PLAYER_NAME']:20s} ({row['team_abbrev']:3s}) | "
              f"Actual: {row['production']:6.1f} | Expected: {row['expected_production']:6.1f} | "
              f"Diff: {row['residual']:+.1f}")

    print(f"\n=== Top {n} Least Accurate Predictions ===")
    print("(Largest absolute residuals - biggest surprises)")

    least_accurate = df_sorted.tail(n).iloc[::-1]
    for _, row in least_accurate.iterrows():
        print(f"  {row['PLAYER_NAME']:20s} ({row['team_abbrev']:3s}) | "
              f"Actual: {row['production']:6.1f} | Expected: {row['expected_production']:6.1f} | "
              f"Diff: {row['residual']:+.1f}")


def print_accuracy_by_salary_range(residuals_df):
    """
    Show model accuracy across different salary ranges.

    Args:
        residuals_df: DataFrame with predictions and actuals
    """
    print("\n=== Accuracy by Salary Range ===")

    # Define salary bins (approximate draft pick ranges)
    df = residuals_df.copy()
    df['salary_bin'] = pd.cut(df['salary'],
                               bins=[0, 2e6, 4e6, 8e6, 15e6],
                               labels=['Second Round', 'Late First', 'Mid First', 'Lottery'])

    for salary_range in ['Lottery', 'Mid First', 'Late First', 'Second Round']:
        subset = df[df['salary_bin'] == salary_range]
        if len(subset) == 0:
            continue

        mae = np.abs(subset['residual']).mean()
        rmse = np.sqrt((subset['residual']**2).mean())

        print(f"  {salary_range:15s} ({len(subset):2d} rookies): MAE = {mae:.1f}, RMSE = {rmse:.1f}")


import pandas as pd


def validate_specific_players(residuals_df, player_names, historical_df=None):
    """
    Show detailed breakdown for specific players to validate residuals.

    Args:
        residuals_df: DataFrame with all rookie data
        player_names: List of player names to check
    """
    print("\n=== Player-by-Player Validation ===")

    for player_name in player_names:
        # Find player (case-insensitive partial match)
        matches = residuals_df[residuals_df['PLAYER_NAME'].str.contains(player_name, case=False, na=False)]

        if matches.empty:
            print(f"\n❌ Player not found: {player_name}")
            continue

        player = matches.iloc[0]

        print(f"\n{'='*60}")
        print(f"Player: {player['PLAYER_NAME']}")
        print(f"Team: {player['team_abbrev']}")
        print(f"Draft Pick: #{player['pick']:.0f}")
        print(f"{'='*60}")

        print(f"\n📊 Stats:")
        print(f"  Games Played: {player['GP']:.0f}")
        print(f"  Total Minutes: {player['MIN']:.0f}")
        print(f"  PIE (Impact Rate): {player['PIE']:.3f}")

        print(f"\n💰 Contract:")
        print(f"  4-Year Avg Salary: ${player['salary']:,.0f}")

        print(f"\n📈 Production Analysis:")
        print(f"  Actual Production: {player['production']:.1f}")
        print(f"    (PIE {player['PIE']:.3f} × Minutes {player['MIN']:.0f})")
        print(f"  Expected Production: {player['expected_production']:.1f}")
        print(f"    (Based on historical rookies at ${player['salary']:,.0f} salary)")

        residual = player['residual']
        print(f"\n{'🟢' if residual > 0 else '🔴'} RESIDUAL VALUE: {residual:+.1f}")

        if residual > 0:
            print(f"  ✓ SURPLUS: Producing {abs(residual):.1f} units MORE than expected")
            print(f"  ✓ This rookie is outperforming their contract")
        else:
            print(f"  ✗ DEFICIT: Producing {abs(residual):.1f} units LESS than expected")
            print(f"  ✗ Historical rookies at this salary typically produce more")

        # Context about their pick range
        if player['pick'] <= 3:
            print(f"\n💡 Context: Top-3 picks face EXTREMELY high expectations")
            print(f"   Historical comparisons: LeBron, Duncan, Zion, Luka, etc.")
            print(f"   Even 'good' rookie seasons can show deficits at this salary level")
        elif player['pick'] <= 10:
            print(f"\n💡 Context: Lottery pick expectations are very high")
            print(f"   Expected to be immediate contributors")
        elif player['pick'] <= 30:
            print(f"\n💡 Context: First-round pick expectations are moderate")
        else:
            print(f"\n💡 Context: Second-round picks have low expectations")
            print(f"   Easy to show surplus value at this price point")

        # Show historical comparisons if available
        if historical_df is not None:
            print(f"\n📚 Historical Rookies at Similar Salary:")
            salary = player['salary']
            # Find rookies within 10% of this salary
            similar_salary = historical_df[
                (historical_df['salary'] >= salary * 0.9) &
                (historical_df['salary'] <= salary * 1.1)
            ].sort_values('production', ascending=False)

            if len(similar_salary) > 0:
                print(f"   Found {len(similar_salary)} historical rookies at ${salary:,.0f} (±10%)")
                print(f"   Their production:")
                print(f"     Average: {similar_salary['production'].mean():.1f}")
                print(f"     Median: {similar_salary['production'].median():.1f}")
                print(f"     Range: {similar_salary['production'].min():.1f} - {similar_salary['production'].max():.1f}")

                print(f"\n   Top performers at this salary:")
                for i, (_, hist_player) in enumerate(similar_salary.head(5).iterrows()):
                    if i >= 5:
                        break
                    print(f"     {hist_player['PLAYER_NAME']:20s} ({hist_player['SEASON']}) - {hist_player['production']:.1f}")

                print(f"\n   {player['PLAYER_NAME']}'s production ({player['production']:.1f}) vs. historical average ({similar_salary['production'].mean():.1f})")
                percentile = (similar_salary['production'] < player['production']).mean() * 100
                print(f"   Ranks in {percentile:.0f}th percentile of historical rookies at this salary")

        print(f"\n⚖️ Is This Residual 'Correct'?")
        print(f"   The residual reflects VALUE vs. CONTRACT, not absolute skill.")
        print(f"   Negative residual = contract is expensive relative to production.")
        print(f"   This doesn't mean {player['PLAYER_NAME']} is a 'bad' player!")

        if residual < -50:
            print(f"\n⚠️  Large deficit detected ({residual:.1f})!")
            print(f"   Possible reasons:")
            print(f"     • Historical data includes generational outliers (LeBron, Luka, Zion)")
            print(f"     • Player is injured or limited minutes")
            print(f"     • Player is on bad team with poor supporting cast")
            print(f"     • Rookie adjustment period (common for top picks)")


def compare_to_historical_picks(residuals_df, historical_df, player_name):
    """
    Compare a player to historical rookies at same draft position.

    Args:
        residuals_df: Current season rookies
        historical_df: Historical rookies dataset
        player_name: Player to analyze
    """
    matches = residuals_df[residuals_df['PLAYER_NAME'].str.contains(player_name, case=False, na=False)]

    if matches.empty:
        print(f"\n❌ Player not found: {player_name}")
        return

    player = matches.iloc[0]
    pick = player['pick']

    print(f"\n=== Historical Comparison for {player['PLAYER_NAME']} ===")
    print(f"Draft Pick: #{pick:.0f}")

    # Find historical rookies at similar pick (±2 picks)
    similar_picks = historical_df[
        (historical_df['pick'] >= pick - 2) &
        (historical_df['pick'] <= pick + 2)
    ]

    if len(similar_picks) > 0:
        print(f"\nHistorical rookies at picks #{pick-2:.0f}-#{pick+2:.0f}:")
        print(f"  Sample size: {len(similar_picks)} rookies")
        print(f"  Average production: {similar_picks['production'].mean():.1f}")
        print(f"  Range: {similar_picks['production'].min():.1f} - {similar_picks['production'].max():.1f}")
        print(f"  {player['PLAYER_NAME']}'s production: {player['production']:.1f}")

        percentile = (similar_picks['production'] < player['production']).mean() * 100
        print(f"\n  {player['PLAYER_NAME']} is in the {percentile:.0f}th percentile")
        print(f"  (Better than {percentile:.0f}% of historical picks at this position)")

