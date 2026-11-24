# Heart Attack Prediction from Lifestyle Habits

## 1. Problem & Dataset

We predict prior heart attacks (`HadHeartAttack`: Yes/No) as a binary classification task using 2022 CDC survey data after removing missing values and duplicates.

## 2. Preprocessing

* **Features (11):** Purely lifestyle and general health habits (`GeneralHealth`, `PhysicalHealthDays`, `MentalHealthDays`, `LastCheckupTime`, `PhysicalActivities`, `SleepHours`, `SmokerStatus`, `ECigaretteUsage`, `AgeCategory`, `BMI`, `AlcoholDrinkers`), dropping unrelated columns.
* **Split & Pipeline:** Used an `80/20` stratified train/test split. Inside `make_pipeline`, we fit both `StandardScaler` and `OneHotEncoder(drop="first")` for numerical and catagorical.

## 3. Models

* **Logistic Regression (`C = 0.1`):**
* **Linear SVM (`C = 10.0`):**

## 4. Results & Recommendation

We evaluate **Recall**, **F1-Score**, and **ROC-AUC** alongside confusion matrices (`lifestyle_11cols_comparison.png`):

| Model                      |    Split     |     Accuracy      |     Precision     |      Recall       |     F1-Score      |      ROC-AUC      |
| :------------------------- | :----------: | :---------------: | :---------------: | :---------------: | :---------------: | :---------------: |
| **Logistic Reg (`C=0.1`)** | Train / Test | 0.703 / **0.704** | 0.128 / **0.128** |   0.768 / 0.760   | 0.220 / **0.219** | 0.806 / **0.803** |
| **Linear SVM (`C=10.0`)**  | Train / Test |   0.696 / 0.697   |   0.127 / 0.126   | 0.777 / **0.767** |   0.218 / 0.217   | 0.806 / **0.803** |

* **Comparison:** Train and test scores match closely (no overfitting). Both linear models perform almost identically.
* **Recommendation:** **Logistic Regression**, because it performs better than SVM (slightly).

## 5. Limitations

Survey data is self-reported and lacks clinical lab values (blood pressure, cholesterol, etc.).