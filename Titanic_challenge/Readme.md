# Titanic Challenge - Kaggle

This repository contains my solution to the [Kaggle Titanic Challenge](https://www.kaggle.com/c/titanic). The goal is to predict which passengers survived the Titanic disaster based on features like age, sex, class, etc.

## Project Overview
- **Score**: Achieved 0.78 on the Kaggle leaderboard.
- **Approach**: Used a Random Forest Classifier with feature engineering (e.g., titles from names, family size, deck extraction).
- **Language**: Python

## Files
- `titanic_model.py`: Main script that performs data cleaning, feature engineering, model training, and generates predictions.
- `requirements.txt`: List of Python dependencies.
- `train.csv`, `test.csv`: Training and test datasets (download from Kaggle) 

## How to Run
1. **Download the Dataset**: Get `train.csv` and `test.csv` from [Kaggle Titanic Challenge](https://www.kaggle.com/c/titanic/data).
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
