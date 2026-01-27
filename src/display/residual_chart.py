"""Create residual value bar chart visualization."""

import matplotlib.pyplot as plt
import numpy as np
import os


def create_residual_chart(residuals_df, current_season, output_path=None, figsize=(12, 16), surplus_color='#2ecc71', deficit_color='#e74c3c'):
    """
    Create horizontal bar chart showing residual values.

    Args:
        residuals_df: DataFrame with residuals
        current_season: Current season string (e.g., "2025-26")
        output_path: Path to save the chart (auto-generated)
        figsize: Figure size
        surplus_color: Green for positive residuals 
        deficit_color: Red for negative residuals 

    Returns:
        Figure object
    """
    if output_path is None:
        output_path = f'outputs/{current_season}_residual_bar_chart.png'
    print("\nCreating residual chart...")

    # Sort by residual
    df = residuals_df.sort_values('residual', ascending=True).copy()

    # Create labels with player name and team
    df['label'] = df['PLAYER_NAME'] + ' (' + df['team_abbrev'] + ')'

    # Determine colors based on residual sign
    colors = [surplus_color if r > 0 else deficit_color for r in df['residual']]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create horizontal bar chart
    y_pos = np.arange(len(df))
    bars = ax.barh(y_pos, df['residual'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Add vertical line at x=0 (expected value)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2, label='Expected Value')

    # Set labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['label'], fontsize=10)
    ax.set_xlabel('Residual Value', fontsize=12, fontweight='bold')
    ax.set_title(f'NBA Rookie Contract Value Analysis {current_season}', fontsize=14, fontweight='bold', pad=20)

    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add text annotations for players
    for idx, (i, row) in enumerate(df.iterrows()):
        residual = row['residual']
        # Position text to the right of positive bars, left of negative bars
        if residual > 0:
            ax.text(residual, idx, f'  +{residual:.1f}', va='center', fontsize=8, fontweight='bold')
        else:
            ax.text(residual, idx, f'  {residual:.1f}', va='center', ha='right', fontsize=8, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=surplus_color, label='Surplus (Outperforming)'),
        Patch(facecolor=deficit_color, label='Deficit (Underperforming)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    # Tight layout
    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # Display the chart
    plt.show(block=False)
    plt.pause(0.1)

    return fig


def create_summary_stats(residuals_df):
    """
    Create summary statistics for the residuals.

    Args:
        residuals_df: DataFrame with residuals

    Returns:
        Dictionary with summary stats
    """
    stats = {
        'total_rookies': len(residuals_df),
        'surplus_rookies': len(residuals_df[residuals_df['residual'] > 0]),
        'deficit_rookies': len(residuals_df[residuals_df['residual'] < 0]),
        'max_surplus': residuals_df['residual'].max(),
        'max_deficit': residuals_df['residual'].min(),
        'mean_residual': residuals_df['residual'].mean(),
        'median_residual': residuals_df['residual'].median(),
    }

    print("\n=== Current Season Summary Statistics ===")
    print(f"  Rookies analyzed: {stats['total_rookies']}")
    print(f"  Providing surplus value: {stats['surplus_rookies']}")
    print(f"  Providing deficit value: {stats['deficit_rookies']}")
    print(f"  Maximum surplus: +{stats['max_surplus']:.2f}")
    print(f"  Maximum deficit: {stats['max_deficit']:.2f}")
    print(f"  Mean residual: {stats['mean_residual']:.2f}")
    print(f"  Median residual: {stats['median_residual']:.2f}")

    return stats
