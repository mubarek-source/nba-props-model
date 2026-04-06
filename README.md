NBA Player Props Prediction Model (ULTORN v5.2)
A machine learning model that predicts NBA player prop outcomes for PrizePicks.
What it does

Pulls live player stats using nba_api
Runs logistic regression to calculate hit probability for each prop line
Blends model probability, site projections, and matchup data into a combined score
Sizes bets using Kelly Criterion
Saves results to a tracked Excel sheet with color-coded confidence levels

Tech Stack

Python
nba_api
scikit-learn
pandas, numpy, openpyxl

How to run

Clone the repo
Install dependencies:

pip install nba_api scikit-learn pandas numpy openpyxl

Add your picks to picks_today.xlsx
Run the model:

python ultorn_model.py

Open ultorn_tracker.xlsx and fill in results nightly
