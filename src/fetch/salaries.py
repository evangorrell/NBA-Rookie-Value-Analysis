"""Load and process rookie salary data."""

import pandas as pd
import os


def load_rookie_scale_salaries(season, data_dir='data'):
    """
    Load rookie scale salaries from CSV file and calculate 4-year average.

    Expected CSV format:
    pick,salary_year1,salary_year2,salary_year3,salary_year4

    Args:
        season: Season string like "2025-26"
        data_dir: Directory containing salary data

    Returns:
        DataFrame with columns: pick, salary (4-year average)
    """
    # Find the salary file
    season_file = os.path.join(data_dir, f'rookie_scale_{season}.csv')
    generic_file = os.path.join(data_dir, 'rookie_scale.csv')

    if os.path.exists(season_file):
        filepath = season_file
    elif os.path.exists(generic_file):
        filepath = generic_file
    else:
        raise FileNotFoundError(
            f"Rookie scale salary file not found."
        )

    df = pd.read_csv(filepath)

    # Calculate 4-year average contract value
    if all(col in df.columns for col in ['salary_year1', 'salary_year2', 'salary_year3', 'salary_year4']):
        # NOTE: year 4 is 0 for second-round picks, handle appropriately
        df['salary'] = df[['salary_year1', 'salary_year2', 'salary_year3', 'salary_year4']].mean(axis=1)
    elif 'salary' in df.columns:
        # Already has salary column
        df = df[['pick', 'salary']].copy()
    else:
        raise ValueError("Salary CSV must have 'salary' or 'salary_year1-4' columns")

    # Select final columns
    df = df[['pick', 'salary']].copy()

    return df


def adjust_salary_for_inflation(salary, from_season, to_season, annual_rate=0.02):
    """
    Adjust salary for inflation to compare historical contracts.

    Args:
        salary: Original salary amount
        from_season: Season string like "2019-20"
        to_season: Season string like "2025-26"
        annual_rate: Annual inflation rate (default 2%)

    Returns:
        Inflation-adjusted salary in to_season dollars
    """
    # Extract year from season strings
    from_year = int(from_season.split('-')[0])
    to_year = int(to_season.split('-')[0])

    years_diff = to_year - from_year

    # Apply compound inflation adjustment
    adjusted_salary = salary * ((1 + annual_rate) ** years_diff)

    return adjusted_salary
