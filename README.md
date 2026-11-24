# Heart Attack Prediction from Lifestyle Habits

## Overview
This project predicts whether a person has had a heart attack (`HadHeartAttack`) based on 11 self-reported lifestyle and general health habits from the 2022 CDC Heart Disease dataset:
- `GeneralHealth`, `PhysicalHealthDays`, `MentalHealthDays`, `LastCheckupTime`
- `PhysicalActivities`, `SleepHours`, `BMI`
- `SmokerStatus`, `ECigaretteUsage`, `AlcoholDrinkers`, `AgeCategory`

We compare two machine learning classification models:
1. **Logistic Regression**
2. **Linear SVM**

## Requirements
Make sure you have Python 3 installed along with the required libraries:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

## Project Files
- `cleaning.py` — Removes missing values and duplicate rows from `heart_2022_with_nans.csv` to create `heart_2022_no_nans.csv`.
- `heart_2022_no_nans.csv` — Cleaned dataset used by the models.
- `models.py` — Preprocesses the 11 features, trains Logistic Regression and Linear SVM, prints evaluations and confusion matrices, and makes comparison plots.

## How to Run
Use ```python``` or ```python3``` (depending on your system)
1. **Clean the dataset**  
   Make sure `heart_2022_with_nans.csv` is in the same folder, then run:
   
   ```bash
   python cleaning.py
   ```
   
2. **Train and evaluate the models:**  
   Make sure `heart_2022_no_nans.csv` is in the same folder, then run:
   ```bash
   python models.py
   ```