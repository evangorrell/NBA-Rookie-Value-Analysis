# NBA Rookie Contract Value Analysis

**Which teams are getting the most (or least) value per dollar from their rookies?**

This project compares each rookie's actual production to what **historical rookies at the same salary** typically delivered. It uses a regression model trained on multi-season data (inflation-adjusted) to establish historical performance benchmarks, then calculates how much current rookies deviate from those norms.

**Residual Value = Actual Production − Expected Production (based on historical salary benchmarks)**

- **Green bars** = Surplus value (outperforming historical averages at that salary)
- **Red bars** = Deficit value (underperforming historical averages at that salary)
- **Zero line** = Historical average for rookies at that price point

---

## Key Finding

**Salary alone is a weak predictor of rookie performance** (R² < 30%). 

Draft position determines salary (fixed scale), but draft position doesn't reliably predict actual NBA performance. Most variance comes from a variety of other factors.

**What this means:** The residuals show which rookies are **outperforming or underperforming relative to historical norms** at their salary level, not precise predictions. Think of it as "contract efficiency" benchmarking.

---

## What You Get

### 1. Residual Bar Chart
- **Y-axis:** Rookie names and teams
- **X-axis:** Residual value (surplus or deficit vs. historical average)
- **Sorted:** Highest surplus at top, biggest deficit at bottom
- **Colors:** Green (surplus), red (deficit), vertical zero line (historical average)

### 2. Accuracy Diagnostic Plot
- Scatter plot: Predicted vs. Actual production
- Shows model fit quality (MAE, RMSE, R²)
- Explains why salary is a weak predictor

### 3. Interactive Player Validation
- Option to request detailed breakdowns for specific players
- Shows historical comparisons at similar salary
- Explains context (top picks vs. second-rounders)
- Ranks player in percentile vs. historical rookies at that price

### 4. Exported Data
- `outputs/2025-26_rookies_residuals.csv` - Full dataset with residuals
- `outputs/2025-26_residual_bar_chart.png` - Main visualization
- `outputs/2025-26_accuracy_diagnostic.png` - Model accuracy plot
- `outputs/model.pkl` - Trained model
- `outputs/historical_data_*.pkl` - Cached historical data

---

## How It Works

### Production Metric
**`production = PIE × Minutes`**

- **PIE (Player Impact Estimate):** NBA's all-in-one efficiency metric (can be negative for poor performance)
- **Minutes:** Volume of playing time
- **Result:** Rate × Volume = Total production value

### Model Approach
1. Train **Gradient Boosting Regressor** on historical rookies (2019-2024 by default)
2. Model learns: `production ~ f(salary)`
3. For current rookies, calculate: `residual = actual - expected`
4. Residuals reveal who's outperforming or underperforming their salary benchmark

### Data Sources

**Rookie Stats:**
- `nba_api` (unofficial NBA stats API)
- Regular season data only (playoffs excluded for consistency)
- Minimum 10 games played

**Rookie Identification:**
- Draft board data (picks 1-60 only, excludes undrafted players)
- Matched to season stats via player ID

**Salary Data:**
- `data/rookie_scale.csv` - NBA rookie scale contracts by draft pick
- **4-year average salary** (total cap commitment)
- **Inflation-adjusted** to current season dollars (~2% annually)

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Analysis

```bash
python main.py
```

**The script will:**
1. Load or fetch historical rookie data (for seasons defined in config file)
2. Train regression model on historical benchmarks
3. Fetch current season (2025-26) rookie data
4. Calculate residual values
5. Display bar chart and accuracy diagnostic
6. Prompt for optional player breakdowns
7. Wait for you to close charts (press Enter to exit)

**First run:** Fetches data from NBA API (~30-60 seconds)
**Subsequent runs:** Loads from cache (~5 seconds)

---

## Configuration

Edit `src/config.py` to customize:

```python
START_YEAR = 2019  # First year of training data (2019 by default)

MIN_GAMES_PLAYED = 10  # Minimum games to include rookie

MODEL_TYPE = "gradient_boosting"  # Regression model type
CROSS_VALIDATION_FOLDS = 5  # CV folds for validation

CHART_FIGSIZE = (12, 16)  # Chart dimensions
SURPLUS_COLOR = "#2ecc71"  # Green
DEFICIT_COLOR = "#e74c3c"  # Red
```

---

## Project Structure

```
NBA-Rookie-Contract-Regression-Value/
├── README.md                    
├── requirements.txt          
├── main.py                  
├── data/
│   └── rookie_scale.csv         # NBA rookie salary scale (4-year averages)
├── outputs/                     # Generated files
│   ├── 2025-26_rookies_residuals.csv
│   ├── 2025-26_residual_bar_chart.png
│   ├── 2025-26_accuracy_diagnostic.png
│   ├── historical_data_2019-20_to_2024-25_for_2025-26.pkl
│   └── model.pkl
└── src/
    ├── config.py               
    ├── fetch/
    │   ├── nba_stats.py         # Fetch NBA stats 
    │   ├── rookies.py           # Filter to rookies 
    │   ├── draft.py             # Fetch draft board data
    │   └── salaries.py          # Load & adjust salaries for inflation
    ├── features/
    │   └── build_dataset.py     # Build training/prediction datasets
    ├── model/
    │   ├── train.py             # Train Gradient Boosting model
    │   ├── predict.py           # Calculate residuals
    │   └── diagnostics.py       # Accuracy metrics & player validation
    └── display/
        └── residual_chart.py    # Generate bar chart visualization
```

---

## Cache Management

**Automatic invalidation:** Cache updates when you change:
- Current season (e.g., 2025-26 → 2026-27)
- Historical date range (e.g., START_YEAR = 2019 → 2015)

**Cache filename format:**
`historical_data_{first-season}_to_{last-season}_for_{current-season}.pkl`

**Example:** `historical_data_2019-20_to_2024-25_for_2025-26.pkl`

**Manual refresh (mid-season updates):**
```bash
rm outputs/historical_data_*.pkl
python main.py  # Re-fetches from NBA API
```

---

## Understanding the Results

### Surplus Value (Green Bars)
- Rookie is producing **more** than historical rookies at that salary
- Team is getting **good value** for the contract cost
- Often: high draft picks exceeding expectations, or late picks overperforming

### Deficit Value (Red Bars)
- Rookie is producing **less** than historical rookies at that salary
- Team is getting **poor value** for the contract cost
- Often: high picks struggling, or injuries/adjustment periods

### Important Context

**Negative residual ≠ Bad player**
- Top-3 picks face extremely high expectations
- Even "good" rookie seasons can show deficits at that salary level
- Contract cost is very high relative to typical performance

**Salary is a weak predictor (by design)**
- Draft position determines salary (fixed scale)
- Draft position doesn't reliably predict performance
- That's why teams seek "value picks" and avoid "busts"

---

## Limitations

1. **Salary-only model:** Doesn't include college stats, combine results, or other predictors
2. **Historical comparison:** Assumes current rookies face similar contexts as past rookies
3. **Small sample:** 48 current rookies vs. 285+ historical rookies
4. **Regular season only:** Excludes playoffs (intentional for consistency)
5. **Single metric:** PIE × Minutes doesn't capture intangibles or defensive impact fully

---

## License

MIT License - Feel free to use and modify

## Credits

Data from [nba_api](https://github.com/swar/nba-api) (unofficial NBA stats wrapper)