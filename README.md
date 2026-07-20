# NBA Player Analytics Model

A Python-based analytics project that analyzes NBA player statistics and historical performance trends to generate player prop prediction scores.

The project uses NBA data, statistical features, and machine learning to evaluate player performance against selected prop lines.

## What It Does

- Retrieves player game logs using `nba_api`
- Retrieves team defensive rankings for matchup analysis
- Processes player statistics including points, rebounds, assists, and three-pointers
- Creates features using recent performance trends, usage, rest days, and opponent data
- Uses logistic regression to generate probability estimates
- Combines model probability with input projection differences to create confidence scores
- Exports prediction results into Excel for tracking

## Built With

- Python
- nba_api
- pandas
- NumPy
- scikit-learn
- openpyxl

## How It Works

1. Player selections are loaded from a user-provided Excel input file.
2. The program retrieves historical player statistics and matchup information.
3. Statistics are processed into model features.
4. A logistic regression model generates probability estimates.
5. Results are ranked using combined confidence scores and saved into an Excel tracker.

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
