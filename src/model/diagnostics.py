"""Model diagnostics - player validation and text-based reporting."""

import pandas as pd
from .. import config


def validate_specific_players(residuals_df, player_names, historical_df=None):
    """
    Show detailed breakdown for specific players to validate residuals.

    Args:
        residuals_df: DataFrame with all rookie data
        player_names: List of player names to check
    """
    for player_name in player_names:
        # Find player
        matches = residuals_df[residuals_df['PLAYER_NAME'].str.contains(player_name, case=False, na=False)]

        if matches.empty:
            print(f"\n{'='*60}")
            print(f"=== {player_name} Validation ===")
            print(f"{'='*60}")
            print(f"  Player not found: {player_name}")
            continue

        player = matches.iloc[0]

        print(f"{player['PLAYER_NAME']}")
        print(f"Team: {player['team_abbrev']}")
        print(f"Draft Pick: #{player['pick']:.0f}")

        print(f"\n  Stats:")
        print(f"  Games Played: {player['GP']:.0f}")
        print(f"  Total Minutes: {player['MIN']:.0f}")
        print(f"  Player Impact Estimate (PIE): {player['PIE']:.3f}")

        print(f"\n  Contract:")
        print(f"  4-Year Avg. Salary: ${player['salary']:,.0f}")

        print(f"\n  Production Analysis:")
        print(f"  Actual Production: {player['production']:.1f}")
        print(f"    (PIE {player['PIE']:.3f} × Minutes {player['MIN']:.0f})")
        print(f"  Expected Production: {player['expected_production']:.1f}")
        print(f"    (Based on historical rookies at ${player['salary']:,.0f} salary)")

        residual = player['residual']
        print(f"\n{'🟢' if residual > 0 else '🔴'} Residual Value: {residual:+.1f}")

        if residual > 0:
            print(f"     SURPLUS: Producing {abs(residual):.1f} units more than expected")
            print(f"     This rookie is outperforming their contract")
        else:
            print(f"     DEFICIT: Producing {abs(residual):.1f} units less than expected")
            print(f"     Historical rookies at this salary typically produce more")

        # Context about their pick range
        if player['pick'] <= 3:
            print(f"\n  Top-3 picks face extremely high expectations")
            print(f"  Even great rookie seasons can show deficits at this salary level")
        elif player['pick'] <= 10:
            print(f"\n  Lottery pick expectations are very high")
            print(f"  Expected to be immediate contributors")
        elif player['pick'] <= 30:
            print(f"\n  First-round pick expectations are moderate")
        else:
            print(f"\n  Second-round picks have low expectations")
            print(f"  Easy to show surplus value at this price point")

        # Show historical comparisons if available
        if historical_df is not None:
            print(f"\n  Historical Rookies at Similar Salary:")
            salary = player['salary']
            # Find rookies within 5% of this salary
            similar_salary = historical_df[
                (historical_df['salary'] >= salary * 0.95) &
                (historical_df['salary'] <= salary * 1.05)
            ].sort_values('production', ascending=False)

            if len(similar_salary) > 0:
                print(f"    Found {len(similar_salary)} historical rookies around ${salary:,.0f} (±5%)")
                print(f"    Their production:")
                print(f"      Average: {similar_salary['production'].mean():.1f}")
                print(f"      Median: {similar_salary['production'].median():.1f}")
                print(f"      Range: {similar_salary['production'].min():.1f} - {similar_salary['production'].max():.1f}")

                print(f"\n   Top performers at this salary:")
                for i, (_, hist_player) in enumerate(similar_salary.head(5).iterrows()):
                    if i >= 5:
                        break
                    print(f"     {hist_player['PLAYER_NAME']:20s} ({hist_player['SEASON']}) - {hist_player['production']:.1f}")

                print(f"\n   {player['PLAYER_NAME']}'s production ({player['production']:.1f}) vs. historical average ({similar_salary['production'].mean():.1f})")
                percentile = (similar_salary['production'] < player['production']).mean() * 100

                print(f"   Ranks in {percentile:.0f}th percentile of historical rookies at this salary since {config.START_YEAR}")

        print("\n  Interpreting Residuals")
        print("    The residual reflects contract value, not absolute skill.")
        print("    Negative residual means contract is expensive relative to production, while positive residual means production is exceeding expectation based on contract.")
        print(f"    This doesn't necessarily assess {player['PLAYER_NAME']}'s skills!")

        if residual < -50:
            print(f"\n   Possible reasons for large deficit:")
            print(f"     - Historical data includes generational outliers")
            print(f"     - Player is injured or limited minutes")
            print(f"     - Player is on bad team with poor supporting cast")
            print(f"     - Rookie adjustment period (common for top picks)")
