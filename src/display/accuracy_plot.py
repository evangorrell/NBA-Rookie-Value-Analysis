"""Create accuracy diagnostic scatter plot."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os


def create_prediction_accuracy_plot(residuals_df, current_season, output_dir='outputs'):
    """
    Create scatter plot showing predicted vs actual production.

    Prints Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R^2 (coefficient of determination)

    Args:
        residuals_df: DataFrame with actual, expected, and residual values
        current_season: Current season string
        output_dir: Directory to save plot
    """
    actual = residuals_df['production'].values
    predicted = residuals_df['expected_production'].values

    # Calculate accuracy metrics
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    print(f"\n  Accuracy metrics:")
    print(f"    Mean Absolute Error (MAE): {mae:.2f}")
    print(f"    Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"    R² Score: {r2:.3f}")

    print(f"\n  What these numbers mean:")

    # MAE interpretation
    print(f"\n  MAE = {mae:.1f}")
    print(f"    - On average, predictions are off by {mae:.1f} production units")
    avg_production = actual.mean()
    mae_pct = (mae / avg_production) * 100
    print(f"    - That's about {mae_pct:.1f}% error relative to average production ({avg_production:.1f})")
    if mae < 20:
        print(f"    ->  Excellent accuracy!")
    elif mae < 30:
        print(f"    ->  Good accuracy")
    else:
        print(f"    ->  Moderate accuracy")

    # RMSE interpretation
    rmse_mae_ratio = rmse / mae
    print(f"\n  RMSE = {rmse:.1f}")
    print(f"    - Similar to MAE but penalizes large errors more")
    print(f"    - RMSE/MAE ratio = {rmse_mae_ratio:.2f}")
    if rmse_mae_ratio < 1.15:
        print(f"    ->  Errors are consistent (few outliers)")
    elif rmse_mae_ratio < 1.4:
        print(f"    ->  Some outlier predictions exist")
    else:
        print(f"    ->  Many large outlier errors (injuries/breakouts?)")

    # R² interpretation
    print(f"\n  R² = {r2:.3f} ({r2*100:.1f}%)")

    if r2 < 0.3:
        print(f"\n  Key finding: Salary alone is a weak predictor of rookie performance")
        print(f"    - Salary explains only {max(0, r2*100):.1f}% of variance")
        print(f"    - The remaining {100 - max(0, r2*100):.1f}% is due to other factors")
        print(f"\n  What this means for the analysis:")
        print(f"    - Residuals show how rookies compare to historical averages at their salary")
        print(f"    - Not precise predictions, but useful benchmarks")
        print(f"    - Surpluses/deficits reflect deviations from typical performance")
        print(f"    - Interpretation: 'Better/worse than historical rookies at this price'")
        print(f"\n  This tool evaluates production value relative to historical rookies")
        print(f"  at the same salary, NOT absolute predictions of future performance.")
    elif r2 > 0.7:
        print(f"    - Salary model explains {r2*100:.1f}% of variance in rookie production")
        print(f"    ->  Excellent! Top-tier for sports analytics")
    elif r2 > 0.6:
        print(f"    - Salary model explains {r2*100:.1f}% of variance in rookie production")
        print(f"    ->  Salary is a good predictor.")
    elif r2 > 0.5:
        print(f"    - Salary model explains {r2*100:.1f}% of variance in rookie production")
        print(f"    ->  Salary is a moderate predictor.")
    else:
        print(f"    - Salary model explains {r2*100:.1f}% of variance in rookie production")
        print(f"    ->  Salary is a weak predictor.")

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

    # Display (non-blocking)
    plt.show(block=False)
    plt.pause(0.1)

    return fig
