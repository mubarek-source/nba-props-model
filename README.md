# NBA Player Analytics Model

A Python-based analytics project that analyzes NBA player statistics and historical performance trends to generate player prop prediction scores.

The project uses NBA data, statistical features, and machine learning techniques to evaluate player performance against selected prop lines.

## What It Does

- Retrieves player game logs using `nba_api`
- Retrieves team defensive rankings for matchup analysis
- Processes player statistics including points, rebounds, assists, and three-pointers
- Creates features using recent performance trends, usage, rest days, and opponent data
- Uses logistic regression to generate probability estimates
- Combines model probability and projection differences into confidence scores
- Exports prediction results into an Excel tracker

## Built With

- Python
- nba_api
- pandas
- NumPy
- scikit-learn
- openpyxl

## How It Works

1. Player selections are loaded from an Excel input file.
2. The program retrieves historical NBA statistics and matchup data.
3. Player performance features are created from recent trends.
4. A logistic regression model generates probability estimates.
5. Results are ranked using combined confidence scores and saved into an Excel tracker.

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
