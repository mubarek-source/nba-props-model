# NBA Player Analytics Model

A Python-based NBA analytics project that analyzes player performance trends and generates prediction scores using historical game data.

The project uses NBA data, statistical features, and machine learning to evaluate player performance patterns and create probability-based predictions.

## What It Does

- Retrieves NBA player game logs using `nba_api`
- Retrieves team defensive statistics for matchup analysis
- Processes player statistics including:
  - Points
  - Rebounds
  - Assists
  - Three-pointers made
- Creates features using:
  - Recent performance trends
  - Minutes played
  - Usage indicators
  - Rest days
  - Opponent defensive rankings
- Uses logistic regression to generate probability estimates
- Combines model probability with projection differences to create ranking scores
- Exports prediction results into Excel for tracking and review

## Built With

- Python
- nba_api
- pandas
- NumPy
- scikit-learn
- openpyxl

## How It Works

1. Player selections are loaded from an Excel input file.
2. The program retrieves historical player statistics from the NBA API.
3. Data is processed into model features.
4. A logistic regression model analyzes historical performance patterns.
5. Results are scored and ranked based on model output.
6. Results are exported into an Excel tracker.

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
