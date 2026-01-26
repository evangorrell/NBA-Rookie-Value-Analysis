"""Configuration for NBA Rookie Contract Value Analysis."""

from datetime import datetime


def get_current_season():
    """
    Automatically determine the current NBA season.
    NBA seasons run from October to June, so:
    - Oct-Dec: current year is the start year (e.g., Oct 2025 -> 2025-26)
    - Jan-Sep: previous year is the start year (e.g., Jan 2026 -> 2025-26)

    Returns:
        Season string like "2025-26"
    """
    now = datetime.now()
    year = now.year
    month = now.month

    # If we're in Jan-Sep, the season started last year
    if month < 10:
        start_year = year - 1
    else:
        start_year = year

    end_year = start_year + 1
    return f"{start_year}-{str(end_year)[-2:]}"


def generate_historical_seasons(current_season, start_year=2019):
    """
    Generate list of historical seasons from start_year up to (but not including) current season.

    Args:
        current_season: Current season string like "2025-26"
        start_year: First year to include in historical data (default: 2019)

    Returns:
        List of season strings like ["2019-20", "2020-21", ...]
    """
    current_start_year = int(current_season.split('-')[0])
    seasons = []

    for year in range(start_year, current_start_year):
        end_year = year + 1
        seasons.append(f"{year}-{str(end_year)[-2:]}")

    return seasons


# Current season to analyze (auto-detected)
CURRENT_SEASON = get_current_season()

# Seasons to include in historical training data (auto-generated)
HISTORICAL_SEASONS = generate_historical_seasons(CURRENT_SEASON, start_year=2019)

# We use PIE (Player Impact Estimate) * Minutes as our production measure
TARGET_METRIC = "PIE"  # Player Impact Estimate
VOLUME_METRIC = "MIN"  # Minutes played

# Minimum games played to include a rookie in analysis
MIN_GAMES_PLAYED = 10

# Model settings
MODEL_TYPE = "gradient_boosting"  # Best option for non-linear relationships
CROSS_VALIDATION_FOLDS = 5 # k = 5

# Visualization settings
CHART_FIGSIZE = (12, 16)
SURPLUS_COLOR = "#2ecc71"  # Green
DEFICIT_COLOR = "#e74c3c"  # Red
